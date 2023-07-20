from copy import deepcopy
from datetime import timedelta
import json
from typing import Any, Dict
import cv2
import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange, repeat
import os

from torch.nn.utils.rnn import pad_sequence
from torchvision.utils import make_grid

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.cldm import ControlLDM
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from dataset import COCODataset, COCODataLoader
from ldm.modules.attention import default
from ldm.modules.diffusionmodules.util import torch_cat_nested
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import log_txt_as_img

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


class CustomTextConditioning():
    def __init__(self, embs, layer_ids=None, token_ids=None, global_prompt_mask=None, global_bos_eos_mask=None):
        """
        embs:       (batsize, seqlen, embdim)
        layer_ids:  (batsize, seqlen) integers, with 0 for no-layer global tokens
        token_ids:  (batsize, seqlen) integers for tokens from tokenizer
        global_prompt_mask:  (batsize, seqlen) bool that is 1 where the global prompt is and 0 where the local regional prompts are
        global_bos_eos_mask: (batsize, seqlen) bool that is 1 where the global bos and eos tokens are and 0 elsewhere
        """
        self.embs = embs
        self.device = self.embs.device
        self.layer_ids = layer_ids
        self.token_ids = token_ids
        self.global_prompt_mask = global_prompt_mask
        self.global_bos_eos_mask = global_bos_eos_mask
        self.cross_attn_masks = None
        self.progress = None
        self.strength = 10
        self.threshold = None
        self.softness = 0.2
        self.controlonly = False
        self.controlledonly = False
        
    def cross_attention_control(self, sim, numheads=1):
        """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
        wf = self.weight_func(sim)
        
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        sim = sim + wf
        return sim
    
    def weight_func(self, sim):
        mask = self.cross_attn_masks[sim.shape[1]].to(sim.dtype)
        ret = mask * sim.std() * self.strength
        return ret
    
    def flatten_inputs_for_gradient_checkpoint(self):
        flat_out = [self.embs]
        def recon_f(x:list):
            self.embs = x[0]
            return self
        return flat_out, recon_f
    
    def torch_cat_nested(self, other):
        # concatenate all torch tensors along batch dimension
        ret = deepcopy(self)
        batsize = self.embs.shape[0]
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == batsize:       # probably concatenatable tensor
                setattr(ret, k, torch_cat_nested(getattr(self, k), getattr(other, k)))
        # ret.embs = torch_cat_nested(self.embs, other.embs)
        # ret.layer_ids = torch_cat_nested(self.layer_ids, other.layer_ids)
        # ret.token_ids = torch_cat_nested(self.token_ids, other.token_ids)
        # ret.global_bos_eos_mask = torch_cat_nested(self.global_bos_eos_mask, other.global_bos_eos_mask)
        # ret.global_prompt_mask = torch_cat_nested(self.global_prompt_mask, other.global_prompt_mask)
        ret.cross_attn_masks = torch_cat_nested(self.cross_attn_masks, other.cross_attn_masks)
        # ret.progress = torch_cat_nested(self.progress, other.progress)
        return ret
    
    
class TokenTypeEmbedding(torch.nn.Module):
    def __init__(self, embdim):
        super().__init__()
        self.emb = torch.nn.Embedding(5, embdim)
        self.merge = torch.nn.Sequential(
            torch.nn.Linear(embdim, embdim//2),
            torch.nn.GELU(),
            torch.nn.Linear(embdim//2, embdim)
        )
        self.gateA = torch.nn.Parameter(torch.randn(embdim) * 1e-3)
        self.gateB = torch.nn.Parameter(torch.randn(embdim) * 1e-3)
        
    def forward(self, tokentypes, contextemb):
        tokentypeemb = self.emb(tokentypes.clamp_min(0))
        ret = self.merge(contextemb + tokentypeemb)
        ret = tokentypeemb * self.gateA + ret * self.gateB
        return ret
    
    
class ProgressEmbedding(torch.nn.Module):
    def __init__(self, embdim) -> None:
        super().__init__()
        self.progress_emb = torch.nn.Sequential(
            torch.nn.Linear(1, embdim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(embdim//2, embdim)
        )
        self.merge = torch.nn.Sequential(
            torch.nn.Linear(embdim, embdim//2),
            torch.nn.GELU(),
            torch.nn.Linear(embdim//2, embdim)
        )
        self.gateA = torch.nn.Parameter(torch.randn(embdim) * 1e-3)
        self.gateB = torch.nn.Parameter(torch.randn(embdim) * 1e-3)
        
    def forward(self, progress, queries):
        progressemb = self.progress_emb(progress)
        ret = self.merge(progressemb + queries)
        ret = progressemb * self.gateA + ret * self.gateB
        return ret
    
    
class CustomCrossAttention(nn.Module):
    # DONE: add model extension to be able to tell where is global and local parts of the prompt
        
    def init_extra(self):
        # conditioning on token type (global BOS, global or local)
        self.token_type_emb = TokenTypeEmbedding(self.to_k.in_features)
        # conditioning on progress (0..1)
        self.progress_emb = ProgressEmbedding(self.to_q.in_features)

    @classmethod
    def from_base(cls, m):
        m.__class__ = CustomCrossAttention
        m.init_extra()
        return m

    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        if context is not None:
            context = context["c_crossattn"][0]
            contextembs = context.embs
        else:
            assert False        # this shouldn't be used as self-attention
            contextembs = x
            
        typeemb = self.token_type_emb(context.captiontypes, contextembs)
        progressemb = self.progress_emb(context.progress[:, None, None], x)

        q = self.to_q(x + progressemb)
        k = self.to_k(contextembs + typeemb)
        v = self.to_v(contextembs)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            sim = einsum('b i d, b j d -> b i j', q, k)
        
        del q, k

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if context is not None:     # cross-attention
            sim = context.cross_attention_control(sim, numheads=h)
        
        # attention
        sim = (sim * self.scale).softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class ControlPWWLDM(ControlLDM):
    first_stage_key = 'image'
    cond_stage_key = 'all'
    control_key = 'cond_image'
    
    # @torch.no_grad()
    # def get_input(self, batch, k, bs=None, *args, **kwargs):
    #     # takes a batch and outputs image x and conditioning info c  --> keep unchanged
    
    def get_learned_conditioning(self, cond):
        # takes conditioning info (cond_key) and preprocesses it to later be fed into LDM
        # returns CustomTextConditioning object
        # called from get_input()
        # must be used with cond_key = "all", then get_input() passes the batch as-is in here
        # DONE: unpack texts, embed them, pack back up and package with cross-attention masks
        
        # 1. unpack texts for encoding using text encoder
            # # below is parallelized implementation
            # pad_token_id = self.cond_stage_model.tokenizer.pad_token_id
            # padmask = cond["layerids"] >= 0
            # numregionsperexample = torch.max(cond["layerids"], 1)[0] + 1
            # offsets = numregionsperexample.cumsum(0)
            # offsets2 = torch.zeros_like(offsets)
            # offsets2[1:] = offsets[:-1]
            # layerids = cond['layerids'] + offsets2[:, None]
            # flatlayerids = torch.masked_select(layerids, padmask)
            # uniquelayerids, uniquelayerids_reverse, uniquelayercounts = torch.unique(flatlayerids, return_inverse=True, return_counts=True)
            
            # token_ids = pad_token_id * torch.ones(len(uniquelayercounts), max(uniquelayercounts), dtype=cond["caption"].dtype, device=cond["caption"].device)
            # flattokenids = torch.masked_select(cond["caption"], padmask)
            # token_ids_scatter_mask = torch.arange(0, max(uniquelayercounts), device=padmask.device)
            # token_ids_scatter_mask = token_ids_scatter_mask < uniquelayercounts[:, None]
            
            # token_ids.masked_scatter_(token_ids_scatter_mask, flattokenids)
        # this is a non-parallelized implementation
        with torch.no_grad():
            pad_token_id = self.cond_stage_model.tokenizer.pad_token_id
            device = cond["caption"].device
            tokenids = cond["caption"].cpu()
            layerids = cond["layerids"].cpu()
            input_ids = []
            for i in range(len(tokenids)):
                start_j = 0
                for j in range(len(tokenids[0])):
                    layerid = layerids[i, j].item()
                    next_layerid = layerids[i, j+1].item() if j+1 < len(tokenids[0]) else -1
                    if next_layerid == -1:
                        break
                    else:     # not padded
                        if next_layerid > layerid:
                            assert next_layerid - layerid == 1
                            input_ids.append(tokenids[i, start_j:j+1])
                            start_j = j+1
                input_ids.append(tokenids[i, start_j:j+1])
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id).to(device)
             
            # 2. encode using text encoder
            outputs = self.cond_stage_model.transformer(input_ids=input_ids, output_hidden_states=self.cond_stage_model.layer=="hidden")
            if self.cond_stage_model.layer == "last":
                text_emb = outputs.last_hidden_state
            elif self.cond_stage_model.layer == "pooled":
                text_emb = outputs.pooler_output[:, None, :]
            else:
                text_emb = outputs.hidden_states[self.layer_idx]
                
            # 3. pack text embs back to original format, ensure compatibility with masks
            out_emb = torch.zeros(tokenids.shape[0], tokenids.shape[1], text_emb.shape[2], dtype=text_emb.dtype, device=text_emb.device)
            tokenids_recon = pad_token_id * torch.ones_like(tokenids)
            k = 0
            for i in range(len(tokenids)):
                start_j = 0
                for j in range(len(tokenids[0])):
                    layerid = layerids[i, j].item()
                    next_layerid = layerids[i, j+1].item() if j+1 < len(tokenids[0]) else -1
                    if next_layerid == -1:
                        break
                    else:     # not padded
                        if next_layerid > layerid:
                            assert next_layerid - layerid == 1
                            tokenids_recon[i, start_j:j+1] = input_ids[k, :j+1-start_j]
                            out_emb[i, start_j:j+1, :] = text_emb[k, :j+1-start_j, :]
                            start_j = j+1
                            k += 1
                tokenids_recon[i, start_j:j+1] = input_ids[k, :j+1-start_j]
                out_emb[i, start_j:j+1, :] = text_emb[k, :j+1-start_j, :]
                k += 1
            assert torch.all(tokenids == tokenids_recon)
            
        global_prompt_mask = cond["captiontypes"] < 2
        global_bos_eos_mask = cond["captiontypes"] == 0     # TODO: fix this (in dataset.py)
        
        ret = CustomTextConditioning(embs=out_emb,
                                     layer_ids=layerids,
                                     token_ids=tokenids,
                                     global_prompt_mask=global_prompt_mask,
                                     global_bos_eos_mask=global_bos_eos_mask)
        
        ret.captiontypes = cond["captiontypes"]
        
        cross_attn_masks = cond["regionmasks"]    
        cross_attn_masks = {res[0] * res[1]: mask.view(mask.size(0), mask.size(1), -1).transpose(1, 2) for res, mask in cross_attn_masks.items() if res[0] <= 64}
        ret.cross_attn_masks = cross_attn_masks
        return ret

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        
        # attach progress to cond["c_crossattn"]        # TODO: check that "t" is a tensor of one value per example in the batch
        cond["c_crossattn"][0].progress = 1 - t / self.num_timesteps
        # cond["c_crossattn"][0].sigma_t = self.sigmas[t]

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, control=None, only_mid_control=self.only_mid_control)
        else:
            # cond["c_crossattn"].on_before_control()
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            if self.global_average_pooling:
                control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
            # cond["c_crossattn"].on_before_controlled()
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, control=control, only_mid_control=self.only_mid_control)
        return eps
    
    def get_trainable_parameters(self):
        params = []
        # select query and key projections as well as new modules from CustomCrossAttention
        for module in self.modules():
            if isinstance(module, CustomCrossAttention):
                params += list(module.to_q.parameters())
                params += list(module.to_k.parameters())
                params += list(module.token_type_emb.parameters())
                params += list(module.progress_emb.parameters())
                
        for param in params:
            param.store_param = True
            
        saved_param_names = set()
        for paramname, param in self.named_parameters():
            if hasattr(param, "store_param") and param.store_param:
                saved_param_names.add(paramname)
        
        return params, saved_param_names
    
    def configure_optimizers(self):
        lr = self.learning_rate
        
        params, _ = self.get_trainable_parameters()
        
        for p in self.parameters():
            p.requires_grad = False
        for p in params:
            p.requires_grad = True
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        _, trainable_param_names = self.get_trainable_parameters()
        print(f"Number of parameters in checkpoint: {len(checkpoint['state_dict'])}")
        newstatedict = {}
        for k, v in checkpoint["state_dict"].items():
            if k in trainable_param_names:
                newstatedict[k] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = newstatedict
        print(f"Number of trained parameters in checkpoint: {len(checkpoint['state_dict'])}")
        return checkpoint
        # DONE: filter state dict to save only those parameters that have been trained

    @torch.no_grad()
    def get_uncond_batch(self, batch):
        uncond_cond = deepcopy(batch)      # DONE: change all prompts to "" and re-tokenize
        bos, eos = self.cond_stage_model.tokenizer.bos_token_id, self.cond_stage_model.tokenizer.pad_token_id
        new_caption = [[] for _ in range(batch["caption"].shape[0])]
        new_layerids = [[] for _ in new_caption]
        new_captiontypes = [[] for _ in new_caption]
        device = batch["caption"].device
        caption = batch["caption"].cpu()
        layerids = batch["layerids"].cpu()
        captiontypes = batch["captiontypes"].cpu()
        prev = None
        for i in range(len(caption)):
            for j in range(len(caption[0])):
                cur = caption[i, j].item()
                if cur == bos or (cur == eos and prev != eos):
                    new_caption[i].append(cur)
                    new_layerids[i].append(layerids[i, j].item())
                    new_captiontypes[i].append(captiontypes[i, j].item())
                prev = cur
        maxlen = caption.shape[1]
        for i in range(len(new_caption)):
            while len(new_caption[i]) < maxlen:
            # for j in range(len(new_caption[i]), maxlen):
                new_caption[i].append(eos)
                new_layerids[i].append(-1)
                new_captiontypes[i].append(-1)
                
        uncond_cond["caption"] = torch.tensor(new_caption).to(device)
        uncond_cond["layerids"] = torch.tensor(new_layerids).to(device)
        uncond_cond["captiontypes"] = torch.tensor(new_captiontypes).to(device)
                
        return uncond_cond
    
    @torch.no_grad()
    def log_images(self, batch, N=None, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        N = batch["image"].shape[0]
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        # N = min(z.shape[0], N)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            n_row = min(z.shape[0], n_row)
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uncond_batch = self.get_uncond_batch(batch)
            _, uc = self.get_input(uncond_batch, self.first_stage_key, bs=N)
            uc_cat, uc_cross = uc["c_concat"][0], uc["c_crossattn"][0]
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log
    
    
def convert_model(model):
    model.__class__ = ControlPWWLDM
    model.first_stage_key = "image"
    model.control_key = "cond_image"
    model.cond_stage_key = "all"
    
    # DONE: replace CrossAttentions that are at attn2 in BasicTransformerBlocks with adapted CustomCrossAttention that takes into account cross-attention masks
    for module in model.model.diffusion_model.modules():
        if module.__class__.__name__ == "BasicTransformerBlock":
            assert not module.disable_self_attn
            module.attn2 = CustomCrossAttention.from_base(module.attn2)
            
    for module in model.control_model.modules():
        if module.__class__.__name__ == "BasicTransformerBlock":
            assert not module.disable_self_attn
            module.attn2 = CustomCrossAttention.from_base(module.attn2)
    
    return model


def get_checkpointing_callbacks(interval=6*60*60, dirpath=None):
    print(f"Checkpointing every {interval} seconds")
    interval_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="interval_delta_epoch={epoch}_step={step}",
        train_time_interval=timedelta(seconds=interval),
        save_weights_only=True,
        save_top_k=-1,
    )
    latest_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="latest_all_epoch={epoch}_step={step}",
        monitor="step",
        mode="max",
        train_time_interval=timedelta(minutes=10),
        save_top_k=1,
    )
    return [interval_checkpoint, latest_checkpoint]


def create_controlnet_pww_model(basemodelname="v1-5-pruned.ckpt", model_name='control_v11p_sd15_seg'):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(f'./models/{model_name}.yaml').cpu()
    # load main weights
    model.load_state_dict(load_state_dict(f'./models/{basemodelname}', location='cpu'), strict=False)
    # load controlnet weights
    model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cpu'), strict=False)
    model.base_model_name = basemodelname
    model.controlnet_model_name = model_name
    
    model = convert_model(model)
    return model


def main(batsize=4,
         version="v1"):
    # Configs
    batch_size = batsize
    logger_freq = 1 #300
    learning_rate = 1e-5
    sd_locked = False
    
    # check dataloader
    ds = COCODataset(split="valid", max_samples=100)
    dl = COCODataLoader(ds, batch_size={384: round(batch_size * 1.2), 448:batch_size, 512: batch_size}, num_workers=batch_size+2)

    model = create_controlnet_pww_model()

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked

    # Misc
    logger = ImageLogger(batch_frequency=logger_freq)
    checkpoints = get_checkpointing_callbacks(interval=60, dirpath=f"coco_checkpoints_{version}/")
    trainer = pl.Trainer(accelerator="gpu", devices=[0], precision=32, 
                         callbacks=checkpoints + [logger])

    # Train!
    trainer.fit(model, dl)
    
    
if __name__ == "__main__":
    main()
    
    # DONE: implement conditioning on which prompt type a token is and also on progress in CustomCrossAttention
    # DONE: implement weight function in CustomTextConditioning
    # DONE: select the right parameters to train (all the CustomCrossAttentions at first)
    # TODO: implement generation of images (log_images())
    #   TODO: adapt get_unconditional_conditioning()
    # DONE: implement checkpointing
    #   DONE: save only the changed parameters
    #   DONE: test checkpointing
    # DONE: how to handle negative prompt? --> replace prompt of all regions and global prompt with ""
    
    
    