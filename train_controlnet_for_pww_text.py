from copy import deepcopy
from datetime import timedelta
import json
import math
import pickle as pkl
from pathlib import Path
import random
import re
from typing import Any, Dict
import cv2
import einops
import fire
import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange, repeat
import os
import gc

from torch.nn.utils.rnn import pad_sequence
from torchvision.utils import make_grid

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.cldm import ControlLDM, ControlNet
from cldm.logger import ImageLogger, nested_to
from cldm.model import create_model, load_state_dict
from dataset import COCOPanopticDataset, COCODataLoader
from ldm.modules.attention import BasicTransformerBlock, default
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, TimestepEmbedSequential
from ldm.modules.diffusionmodules.util import conv_nd, timestep_embedding, torch_cat_nested
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import SeedSwitch, log_txt_as_img, seed_everything

from gradio_pww import _tokenize_annotated_prompt, create_tools, CustomTextConditioning as CTC_gradio_pww

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


class CustomTextConditioning():
    def __init__(self, global_text_cond=None, local_text_cond=None):
        self.global_text_cond, self.local_text_cond = global_text_cond, local_text_cond
    
    def flatten_inputs_for_gradient_checkpoint(self):
        flat_out = [self.global_text_cond, self.local_text_cond]
        def recon_f(x:list):
            self.global_text_cond = x[0]
            self.local_text_cond = x[1]
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
    
    
class ExtraControlLayer(torch.nn.Module):
    def __init__(self, out_dim=320, inner_dim=None, **kw):
        super().__init__(**kw)
        self.out_dim = out_dim
        self.inner_dim = (self.out_dim * 4) if inner_dim is None else inner_dim
        self.ff1 = torch.nn.Conv2d(self.out_dim, self.inner_dim, kernel_size=1)
        self.ff2 = torch.nn.Conv2d(self.inner_dim, self.out_dim, kernel_size=1)
        self.nonlin = torch.nn.SiLU()
        
    def forward(self, x):
        y = self.ff1(x)
        y = self.nonlin(y)
        y = self.ff2(y)
        # y = self.nonlin(y)
        y = y + x
        return y
    
    
class ExtraControlBlock(torch.nn.Module):
    def __init__(self, dim, inner_dim=None, text_dim=768, num_layers=3, **kw):
        super().__init__(**kw)
        self.dim, self.num_layers = dim, num_layers
        self.text_dim = text_dim
        self.inner_dim = (self.dim * 4) if inner_dim is None else inner_dim
        self.init_block = torch.nn.Conv2d(self.text_dim, self.dim, 1)
        self.main_block = torch.nn.Sequential(*[
            ExtraControlLayer(self.dim, inner_dim=self.inner_dim) for i in range(self.num_layers)
        ])
        
    def forward(self, text_hint):
        y = self.init_block(text_hint.permute(0, 3, 1, 2))
        y = self.main_block(y)
        return y
    
    
class ExtendedControlNet(ControlNet):
    def init_extension(self, num_layers=3):
        self.text_hint_block = ExtraControlBlock(dim=self.model_channels, num_layers=num_layers)
    
    def forward(self, x, hint, timesteps, context, text_hint=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)
        guided_text_hint = self.text_hint_block(text_hint)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint + guided_text_hint
                guided_hint, guided_text_hint = None, None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs
    
    

class ControlPWWLDM(ControlLDM):
    first_stage_key = 'image'
    cond_stage_key = 'all'
    control_key = 'cond_image'
    
    # @torch.no_grad()
    # def get_input(self, batch, k, bs=None, *args, **kwargs):
    #     # takes a batch and outputs image x and conditioning info c  --> keep unchanged
    
    def set_control_drop(self, p=0.):
        self.control_drop = p
    
    def encode_using_text_encoder(self, input_ids):
        outputs = self.cond_stage_model.transformer(input_ids=input_ids, output_hidden_states=self.cond_stage_model.layer=="hidden")
        # if self.cond_stage_model.layer == "last":
        #     text_emb = outputs.last_hidden_state
        # elif self.cond_stage_model.layer == "pooled":
        #     text_emb = outputs.pooler_output[:, None, :]
        # else:
        #     text_emb = outputs.hidden_states[self.layer_idx]
        return outputs.last_hidden_state, outputs.pooler_output
    
    def get_learned_conditioning(self, cond):
        # takes conditioning info (cond_key) and preprocesses it to later be fed into LDM
        # returns CustomTextConditioning object
        # called from get_input()
        # must be used with cond_key = "all", then get_input() passes the batch as-is in here
        # DONE: unpack texts, embed them, pack back up and package with cross-attention masks
        
        with torch.no_grad():
            pad_token_id, bos_token_id, modelmaxlen = self.cond_stage_model.tokenizer.pad_token_id, self.cond_stage_model.tokenizer.bos_token_id, self.cond_stage_model.tokenizer.model_max_length 
            device = cond["caption"].device
            tokenids = cond["caption"]
            # layerids = cond["layerids"].cpu()
            encoder_layerids = cond["encoder_layerids"]
            numlocaldescriptions = encoder_layerids.max()+1
            assert tokenids.shape[1] == modelmaxlen * numlocaldescriptions
            tokenids = tokenids.view(tokenids.shape[0], numlocaldescriptions, modelmaxlen)
            encoder_layerids = encoder_layerids.view(tokenids.shape[0], numlocaldescriptions, modelmaxlen)
            encoder_layerids = encoder_layerids[:, :, 0]
            
            # 2. encode using text encoder
            input_ids = tokenids.view(-1, tokenids.shape[-1])
            out_emb, pool_emb = self.encode_using_text_encoder(input_ids)
            text_emb = out_emb.view(tokenids.shape[0], tokenids.shape[1], out_emb.shape[1], out_emb.shape[2])
            
            # 3. extract only the global description
            global_text_cond = text_emb[:, 0]
            local_emb = pool_emb.view(tokenids.shape[0], tokenids.shape[1], out_emb.shape[2])
            
        maskdim = cond["image"].shape[1] // 8
        
        masks = cond["regionmasks"][(maskdim, maskdim)]
        
        newshape = tokenids.shape + masks.shape[-2:]
        masks = masks.view(*newshape)[:, :, 0]
        
        _coverage = torch.zeros_like(masks[:, 0])
        local_index_map = torch.zeros_like(_coverage.long())
        
        for i in range(masks.shape[1]):
            local_index_map.masked_fill_(_coverage < masks[:, i], i)
            if i > 0:
                _coverage = torch.max(_coverage, masks[:, i])
            
        local_text_cond = torch.zeros(*(local_index_map.shape + (local_emb.shape[-1],)), device=local_emb.device, dtype=local_emb.dtype)
        for i in range(local_index_map.shape[0]):
            local_text_cond[i] = local_emb[i][local_index_map[i]]
        
        ret = CustomTextConditioning(global_text_cond=global_text_cond,
                                     local_text_cond=local_text_cond)
        return ret
    
    
    # def get_learned_conditioning(self, c):
    #     return self.encode_using_text_encoder(c)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, control=None, only_mid_control=self.only_mid_control)
        else:  
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, 
                                         context=cond['c_crossattn'][0].global_text_cond, 
                                         text_hint=cond['c_crossattn'][0].local_text_cond)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            if self.global_average_pooling:
                control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
            
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond['c_crossattn'][0].global_text_cond, control=control, only_mid_control=self.only_mid_control)
        return eps
    
    def get_trainable_parameters(self):
        params = []
        saved_param_names = []
        modesplits = self.mode.split("+")
        for paramname, param in self.named_parameters():
            addit = False
            if paramname.startswith("control_model.text_hint_block"):
                addit = True
            if "full" in modesplits:
                if paramname.startswith("control_model"):
                    addit = True
            elif "firstinput" in modesplits or "input:first" in modesplits: # first conv block after merging input and hint
                if paramname.startswith("control_model.input_blocks.1"):
                    addit = True
            elif "hint" in modesplits:
                if paramname.startswith("control_model.input_hint_block"):
                    addit = True
            elif "firsthint" in modesplits or "hint:first" in modesplits:
                if paramname.startswith("control_model.input_hint_block.0"):
                    addit = True
            elif "hint:firstthree" in modesplits:
                if  paramname.startswith("control_model.input_hint_block.0") or \
                    paramname.startswith("control_model.input_hint_block.2") or \
                    paramname.startswith("control_model.input_hint_block.4"):
                    addit = True
            else:
                raise Exception(f"Unknown mode: {self.mode}")
            if addit:
                saved_param_names.append(paramname)
                params.append(param)
        
        return params, set(saved_param_names)
    
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
    def get_uncond_batch(self, batch):      # DONE: change regionmasks to fit new prompts
        uncond_cond = deepcopy(batch)       # DONE: change all prompts to "" and re-tokenize
        bos, eos = self.cond_stage_model.tokenizer.bos_token_id, self.cond_stage_model.tokenizer.pad_token_id
        maxlen = self.cond_stage_model.tokenizer.model_max_length
        bs = batch["caption"].shape[0]
        device = batch["caption"].device
        
        new_caption3 = torch.ones(bs, maxlen, device=device, dtype=torch.long) * eos
        new_caption3[:, 0] = bos
        new_layerids3 = torch.zeros_like(new_caption3)
        new_captiontypes3 = torch.zeros_like(new_caption3) + 1
        new_captiontypes3[:, 0] = 0
        
        new_regionmasks3 = {k: torch.ones(bs, new_caption3.shape[1], v.shape[2], v.shape[3], device=v.device, dtype=v.dtype) 
                            for k, v in batch["regionmasks"].items()}
        
        uncond_cond["caption"] = new_caption3  #torch.tensor(new_caption).to(device)
        uncond_cond["layerids"] = new_layerids3  #torch.tensor(new_layerids).to(device)
        uncond_cond["encoder_layerids"] = new_layerids3.clone()  #torch.tensor(new_layerids).to(device)
        uncond_cond["captiontypes"] = new_captiontypes3  #torch.tensor(new_captiontypes).to(device)
        uncond_cond["regionmasks"] = new_regionmasks3
                
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
        log["reconstruction"] = reconstrimg = self.decode_first_stage(z)  #.clamp(0, 1) * 2.0 - 1.0
        log["control"] = controlimg = c_cat * 2.0 - 1.0
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
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning={"c_concat": [uc_cat], "c_crossattn": [uc_cross]},
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = generated_img = x_samples_cfg
            
        log[f"all"] = torch.cat([reconstrimg, controlimg, generated_img], 2)
        del log["reconstruction"]
        del log["control"]
        del log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"]

        return log
    

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


def create_controlnet_model(basemodelname="v1-5-pruned.ckpt", model_name='control_v11p_sd15_seg', mode="full", loadckpt=""):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(f'./models/{model_name}.yaml').cpu()
    # load main weights
    model.load_state_dict(load_state_dict(f'./models/{basemodelname}', location='cpu'), strict=False)
    # load controlnet weights
    model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cpu'), strict=False)
    model.base_model_name = basemodelname
    model.controlnet_model_name = model_name
    
    model.__class__ = ControlPWWLDM
    model.first_stage_key = "image"
    model.control_key = "cond_image"
    model.cond_stage_key = "all"
    
    model.control_model.__class__ = ExtendedControlNet
    model.control_model.init_extension(num_layers=4)
    
    model.mode = mode
    
    if loadckpt != "":
        print(f"loading trained parameters from {loadckpt}")
        # refparam1a = model.model.diffusion_model.middle_block[1].proj_in.weight.data.clone()
        # refparam2a = deepcopy(model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2l.to_q.weight.data.clone())
        ckpt_state_dict = load_state_dict(loadckpt, location="cpu")
        # testing the partial loading
        model.load_state_dict(ckpt_state_dict, strict=False)
        # refparam1b = model.model.diffusion_model.middle_block[1].proj_in.weight.data.clone()
        # refparam2b = deepcopy(model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2l.to_q.weight.data.clone())
        # assert torch.all(refparam1a == refparam1b)
    return model


class TrainMode():
    def __init__(self, name):
        self.chunks = name.split("+")
        self.basename = self.chunks[0]
        
    @property
    def name(self):
        return "+".join(self.chunks)
    
    @property
    def use_global_prompt_only(self):
        return False
        
    @property
    def augment_global_caption(self):
        return False
        
    @property
    def is_test(self):
        return "test" in self.chunks
        
    @property
    def replace_layerids_with_encoder_layerids(self):
        return True
        
    def addchunk(self, chunk:str):
        self.chunks.append(chunk)
        return self
    
    def __add__(self, chunk:str):
        return self.addchunk(chunk)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __eq__(self, other:str):
        return self.name.__eq__(other)
    
    def __hash__(self):
        return hash(self.name)
    
    def __contains__(self, item):
        return item in self.chunks


def main(batsize=5,
         version="v4.2",
         datadir="/USERSPACE/lukovdg1/coco2017/",
         devexamples="evaldata/extradev1b.pkl", # "extradev.examples.pkl",  #"coco2017.4dev.examples.pkl",
         devices=(2,),
         mode="hint+firstinput",     # "full", "firstinput", "firsthint", "hint"
         numtrain=-1,
         forreal=False,
         seed=12345,        # seed for training
         log_image_seed=42,     # seed for generating logging images
         loadckpt="",
         ):  
    args = locals().copy()
    # print(args)
    print(json.dumps(args, indent=4))     
    print(devices, type(devices), devices[0])
    # Configs
    batch_size = batsize
    logger_freq = 1000 if forreal else 10 #300
    learning_rate = 1e-5
    sd_locked = False
    
    numtrain = None if numtrain == -1 else numtrain
    
    expnr = 1
    def get_exp_name(_expnr):
        ret = f"checkpoints/pretrain-{version}/checkpoints_coco_controlnetmod_extended_{mode}_exp_{_expnr}{'_forreal' if forreal else ''}"
        return ret
        
    exppath = Path(get_exp_name(expnr))
    while exppath.exists():
        expnr += 1
        exppath = Path(get_exp_name(expnr))
        
    trainmode = TrainMode(mode)
        
    # load dev set from pickle
    loadedexamples = []
    for devexamplepath in devexamples.split(","):
        with open(devexamplepath, "rb") as f:
            loadedexamples_e = pkl.load(f)
            loadedexamples += loadedexamples_e
    # override pickled defaults
    valid_ds = COCOPanopticDataset(examples=loadedexamples, max_masks=30, mergeregions=False, casmode=trainmode)
    valid_dl = COCODataLoader(valid_ds, batch_size=4, num_workers=4 if forreal else 0, shuffle=False)
    
    model = create_controlnet_model(mode=mode, loadckpt=loadckpt)
    
    seedswitch = SeedSwitch(seed, log_image_seed)
    image_logger = ImageLogger(batch_frequency=logger_freq, dl=valid_dl, seed=seedswitch)
    
    exppath.mkdir(parents=True, exist_ok=False)
    print(f"logging in {exppath}")
    with open(exppath / "args.json", "w") as f:
        json.dump(args, f, indent=4)
    
    ds = COCOPanopticDataset(maindir=datadir, split="train" if forreal else "valid", max_masks=30, mergeregions=False, casmode=trainmode,
                    max_samples=numtrain if numtrain is not None else (None if forreal else 1000))
    
    print(len(ds))
    batsizes = {384: round(batch_size * 2.4), 448:round(batch_size * 1.4), 512: batch_size}
    print(f"Batch sizes: {batsizes}")
    dl = COCODataLoader(ds, batch_size=batsizes, 
                        num_workers=max(batsizes.values()) if forreal else 0,
                        shuffle=True)

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
        
    checkpointinterval = 30*60
    if "firstinput" in trainmode or "input:first" in trainmode:
        checkpointinterval = 4*60*60
    if "full" in trainmode:
        checkpointinterval = 8*60*60
    checkpoints = get_checkpointing_callbacks(interval=checkpointinterval if forreal else 60*10, dirpath=exppath)
    logger = pl.loggers.TensorBoardLogger(save_dir=exppath)
    
    max_steps = -1
    
    trainer = pl.Trainer(accelerator="gpu", devices=devices, 
                        precision=32, max_steps=max_steps,
                        logger=logger,
                        callbacks=checkpoints + [image_logger])

    # Train!
    print(f"Writing to {exppath}")
    
    # with open(exppath / "args.json", "w") as f:
    #     json.dump(args, f)
    trainer.fit(model, dl)
    
    
if __name__ == "__main__":
    fire.Fire(main)