from cldm.cldm import ControlLDM
from share import *
import config
from functools import partial

import cv2
import einops
import gradio as gr
import math
import numpy as np
import torch
import random
import psd_tools
import re
import torch.nn.functional as F
import pickle
from PIL import Image
import fire
from copy import deepcopy, copy

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
# from annotator.uniformer import UniformerDetector
# from annotator.oneformer import OneformerCOCODetector, OneformerADE20kDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# for custom promptencoder
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

import os
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def _tokenize_annotated_prompt(prompt, tokenizer):
    prompt = re.split(r"(\{[^\}]+\})", prompt)
    _prompt = []
    _layer_id = []
    for e in prompt:
        m = re.match(r"\{(.+):(\d+)\}", e)
        if m:
            _prompt.append(m.group(1))
            _layer_id.append(int(m.group(2)) + 1)
        else:
            _prompt.append(e)
            _layer_id.append(0)

    for i in range(len(_prompt)):
        if i == len(_prompt) - 1:
            tokenized = tokenizer([_prompt[i]],
                                  padding="max_length",
                                  max_length=tokenizer.model_max_length,
                                  return_overflowing_tokens=False,
                                  truncation=True,
                                  return_tensors="pt")
        else:
            tokenized = tokenizer([_prompt[i]], return_tensors="pt")
        _prompt[i] = tokenized.input_ids[0, (0 if i == 0 else 1):(-1 if i < len(_prompt) - 1 else None)]
        _layer_id[i] = torch.tensor([_layer_id[i]]).repeat(len(_prompt[i]))

    token_ids = torch.cat(_prompt, 0)
    token_ids = token_ids[:min(len(token_ids), tokenizer.model_max_length)]
    layer_ids = torch.cat(_layer_id, 0)
    layer_ids = layer_ids[:min(len(layer_ids), tokenizer.model_max_length)]

    assert len(token_ids) <= tokenizer.model_max_length
    return token_ids, layer_ids


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
        self.strength = None
        self.threshold = None
        self.softness = 0.2
        self._post_init()
        
    def _post_init(self):
        pass
        
    def cross_attention_control(self, sim):
        """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
        sim = sim + self.weight_func(sim)
        return sim
    
    def weight_func(self, sim):
        return self._weight_func(self, sim)
    
    def set_method(self, methodstr, **kw):
        if methodstr == "SepMix":
            selfcopy = copy(self)
            self.__class__ = SepMixTextConditioning
            self._localcond = selfcopy
            self._localcond.set_method("Separate")
            self._localcond.threshold = 2
            self._localcond.strength = 5
            self._globalcond = copy(selfcopy)
            self._globalcond.set_method("PosAttn:WithBOS")
            self._globalcond.threshold = 0.85
            self._globalcond.softness = 0.6
            self._globalcond.strength = 5
            return self
        else:
            method_functions = {
                "Ediffi": ediffi_weight_func,
                "Ediffi++": ediffi_pp_weight_func,
                "PosAttn": posattn_weight_func,
                "PosAttn:WithBOS": posattn_with_bos_weight_func,
                "PosAttn:RaiseBOS": posattn_raise_bos_weight_func,
                "Separate": posattn_separate_weight_func,
                "SepSwitch": posattn_sepswitch_weight_func,
                "GlobalNeg": global_neg_weight_func,
            }
            if methodstr in method_functions:
                f = method_functions[methodstr]
                f = partial(f, **kw)
                self._weight_func = f
                return self
            else:
                raise Exception(f"unknown method: {methodstr}")
        
        
class SepMixTextConditioning(CustomTextConditioning):
    def use_local_cond(self):
        self.use_local = True
    
    def use_global_cond(self):
        self.use_local = False
    
    def weight_func(self, sim):
        if self.use_local:
            self._localcond.progress = self.progress
            return self._localcond.weight_func(sim)
        else:
            self._globalcond.progress = self.progress
            return self._globalcond.weight_func(sim)
        
    def mix_eps(self, localeps, globaleps):
        a, b = max(0, self.threshold - self.softness / 2), min(self.threshold + self.softness / 2, 1)
        assert self.strength >= 0. and self.strength <= 1.
        weight = (1 -_threshold_f(self.progress, a, b)) * self.strength
        ret = globaleps * (1 - weight) + weight * localeps
        return ret
        

def ediffi_weight_func(self, sim, sigma_square=False, qk_std=False, **kw):
    # Ediffi uses the global prompt with layer annotations and adds the cross-attention mask to the attention scores
    
    # create negative mask that completely ignores non-global prompts
    negmask = 1 - self.global_prompt_mask[:, None].to(sim.dtype)
    max_neg_value = -torch.finfo(sim.dtype).max
    
    # use settings to compute the strength of the mask
    a = self.strength * math.log(1 + (self.sigma_t **2 if sigma_square else self.sigma_t)) * (sim.std() if qk_std else sim.max())
    
    # get the cross attention mask for the right resolution and multiply with final mask strength
    mask = self.cross_attn_masks[sim.shape[1]]
    ret = mask * a
    ret.masked_fill_(negmask > 0.5, max_neg_value)        # (batsize, 1, seqlen)
    return ret


def ediffi_pp_weight_func(self, sim, sigma_square=True, qk_std=True, **kw):
    return ediffi_weight_func(self, sim, sigma_square=sigma_square, qk_std=qk_std, **kw)


def global_neg_weight_func(self, sim, **kw):
    # only applies global negative mask
    # create negative mask that completely ignores non-global prompts
    negmask = 1 - self.global_prompt_mask[:, None].to(sim.dtype)
    max_neg_value = -torch.finfo(sim.dtype).max
    
    ret = self.cross_attn_masks[sim.shape[1]]
    ret = torch.zeros_like(ret)
    ret.masked_fill_(negmask > 0.5, max_neg_value)
    return ret


def _threshold_f(p, a, b=1): # transitions from 0 at p=a to 1 at p=b using sinusoid curve
    threshold = (a, b)
    b = max(threshold)
    b = min(b, 1)
    a = min(threshold)
    a = min(b, a)
    a = max(a, 0)
    b = max(a, b)
    if p <= a:
        weight = 0
    elif p > a and p < b:
        midpoint = (b - a) / 2 + a
        weight = (math.sin(math.pi * (p - midpoint) / (b - a)) + 1) * 0.5
    else:
        weight = 1
    return weight


def posattn_weight_func(self, sim, **kw):
    # Positive attention with threshold uses the global prompt with layer annotations and adds the cross-attention mask to the attention scores
    
    # create negative mask that completely ignores non-global prompts
    negmask = 1 - self.global_prompt_mask[:, None].to(sim.dtype)
    max_neg_value = -torch.finfo(sim.dtype).max
    
    a, b = max(0, self.threshold - self.softness / 2), min(self.threshold + self.softness / 2, 1)
    
    weight = 1 - _threshold_f(self.progress, a, b)
    
    # get the cross attention mask for the right resolution and multiply with final mask strength
    mask = self.cross_attn_masks[sim.shape[1]]
    ret = mask * weight * self.strength * sim.std()
    ret.masked_fill_(negmask > 0.5, max_neg_value)
    return ret


def posattn_with_bos_weight_func(self, sim, **kw):
    # Positive attention with threshold uses the global prompt with layer annotations and adds the cross-attention mask to the attention scores
    
    # create negative mask that completely ignores non-global prompts
    negmask = 1 - self.global_prompt_mask[:, None].to(sim.dtype)
    max_neg_value = -torch.finfo(sim.dtype).max
    
    a, b = max(0, self.threshold - self.softness / 2), min(self.threshold + self.softness / 2, 1)
    
    weight = 1 - _threshold_f(self.progress, a, b)
    
    # get the cross attention mask for the right resolution and multiply with final mask strength
    mask = self.cross_attn_masks[sim.shape[1]]
    bosmask = self.global_bos_eos_mask[:, None].to(sim.dtype)  # (batsize, 1, seqlen)
    mask = mask + bosmask
    ret = mask * weight * self.strength * sim.std()
    ret.masked_fill_(negmask > 0.5, max_neg_value)
    return ret


def posattn_raise_bos_weight_func(self, sim, **kw):     # sim: (batsize*numheads, numpixel, seqlen)
    # Positive attention with threshold uses the global prompt with layer annotations and adds the cross-attention mask to the attention scores
    # This variant masks all non-relevant regions across all timesteps but brings up the BOS token.
    
    # create negative mask that completely ignores non-global prompts
    negmask = 1 - self.global_prompt_mask[:, None].to(sim.dtype)        # (batsize, 1, seqlen)
    max_neg_value = -torch.finfo(sim.dtype).max
    
    # fixed mask which keeps the relevant regions raised always
    fixedmask = self.cross_attn_masks[sim.shape[1]]     # (batsize, numpixel, seqlen)
    
    # bos-eos mask which is being gradually raised according to schedule
    bosmask = self.global_bos_eos_mask[:, None].to(sim.dtype)  # (batsize, 1, seqlen)
    
    a, b = max(0, self.threshold - self.softness / 2), min(self.threshold + self.softness / 2, 1)
    weight = _threshold_f(self.progress, a, b)
    
    # get the cross attention mask for the right resolution and multiply with final mask strength
    ret = (fixedmask + weight * bosmask) * self.strength * sim.std()    
    ret.masked_fill_(negmask > 0.5, max_neg_value)
    return ret
        

def posattn_separate_weight_func(self, sim, **kw):
    # Positive attention with threshold that uses the local prompts and adds the cross-attention mask to the attention scores
    
    # create negative mask that completely ignores non-global prompts
    negmask = self.global_prompt_mask[:, None].to(sim.dtype)
    max_neg_value = -torch.finfo(sim.dtype).max
    
    a, b = max(0, self.threshold - self.softness / 2), min(self.threshold + self.softness / 2, 1)
    
    weight = 1 - _threshold_f(self.progress, a, b)
    
    # get the cross attention mask for the right resolution and multiply with final mask strength
    mask = self.cross_attn_masks[sim.shape[1]]
    ret = mask * weight * self.strength * sim.std()
    ret.masked_fill_(negmask > 0.5, max_neg_value)
    return ret


def posattn_sepswitch_weight_func(self, sim, **kw):
    # Positive attention with threshold that uses the local prompts 
    # and then switches to global prompt
    
    # fixed mask which keeps the relevant regions raised always
    fixedmask = self.cross_attn_masks[sim.shape[1]]     # (batsize, numpixel, seqlen)
    bosmask = self.global_bos_eos_mask[:, None].to(sim.dtype)  # (batsize, 1, seqlen)
    fixedmask = fixedmask + bosmask
    
    # get the cross attention mask for the right resolution and multiply with final mask strength
    ret = fixedmask * self.strength * sim.std()
    
    a, b = max(0, self.threshold - self.softness / 2), min(self.threshold + self.softness / 2, 1)
    weight = 1 - _threshold_f(self.progress, a, b)
    
    # create negative mask that completely ignores non-global prompts
    globalmask = self.global_prompt_mask[:, None].to(sim.dtype)
    max_neg_value = -torch.finfo(sim.dtype).max
    if self.progress <= a:
        ret.masked_fill_(globalmask > 0.5, max_neg_value)
    elif self.progress >= b:
        ret.masked_fill_(globalmask <= 0.5, max_neg_value)
    else:
        global_ret = ret.clone()
        global_ret.masked_fill_(globalmask <= 0.5, 0)
        local_ret = ret.clone()
        local_ret.masked_fill_(globalmask > 0.5, 0)
        ret = local_ret * weight + (1 - weight) * global_ret
    return ret


class CustomCLIPTextEmbedder(FrozenCLIPEmbedder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    def forward(self, fullprompt):
        fullprompt, batsize = fullprompt if isinstance(fullprompt, tuple) else (fullprompt, 1)
        
        layer_idses = []
        text_embeddingses = []
        token_idses = []
        
        layertexts = fullprompt.split("\n|")
        
        # encode layers separately
        for i, pos_prompt in enumerate(layertexts):
            if i == 0:
                token_ids, layer_ids = _tokenize_annotated_prompt(pos_prompt.strip(), self.tokenizer)
                global_len = token_ids.shape[0]
                global_bos_pos = 0
                global_eos_pos = torch.nonzero(token_ids == self.tokenizer.eos_token_id)[0][0]
            else:
                pos_prompt = pos_prompt.strip()
                token_ids = self.tokenizer([pos_prompt], return_tensors="pt",
                                        max_length=self.max_length, return_overflowing_tokens=False,
                                        truncation=True)["input_ids"][0]
                layer_ids = torch.tensor([i] * token_ids.shape[0])
            outputs = self.transformer(input_ids=token_ids[None].to(self.device), output_hidden_states=self.layer=="hidden")
            if self.layer == "last":
                z = outputs.last_hidden_state
            elif self.layer == "pooled":
                z = outputs.pooler_output[:, None, :]
            else:
                z = outputs.hidden_states[self.layer_idx]
                
            layer_idses.append(layer_ids)
            token_idses.append(token_ids)
            text_embeddingses.append(z)
            
        layer_ids = torch.cat(layer_idses, 0)[None].repeat(batsize, 1)
        token_ids = torch.cat(token_idses, 0)[None].repeat(batsize, 1)
        text_embeddings = torch.cat(text_embeddingses, 1).repeat(batsize, 1, 1)
        global_prompt_mask = torch.zeros_like(token_ids)
        global_bos_eos_mask = torch.zeros_like(global_prompt_mask)
        global_prompt_mask[:, :global_len] = 1
        global_bos_eos_mask[:, global_bos_pos] = 1
        global_bos_eos_mask[:, global_eos_pos] = 1
        
        ret = CustomTextConditioning(text_embeddings, layer_ids.to(self.device), token_ids.to(self.device),
                                     global_prompt_mask.to(self.device), global_bos_eos_mask.to(self.device))
        return ret
        # return text_embeddings, layer_ids.to(self.device), token_ids.to(self.device)


def _img_importance_flatten(img: torch.tensor, w: int, h: int) -> torch.tensor:
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        # scale_factor=1 / ratio,
        size=(w, h),
        mode="bilinear",
        align_corners=True,
    ).squeeze()


def unpack_layers(layers):
    # Unpacks the layers PSD file, creates masks, preprocesses prompts etc.
    unpacked_layers = []
    global_descriptions = {}
    for layer in layers:
        if layer.name == "Background":      # Ignore layer named "Background"
            continue
        if layer.name.startswith("[GLOBAL]"):
            assert len(global_descriptions) == 0, "Only one global description"
            splits = [x.strip() for x in layer.name[len("[GLOBAL]"):].split("|")]
            pos, neg = splits if len(splits) == 2 else (splits[0], "")
            global_descriptions["pos"] = pos
            global_descriptions["neg"] = neg
            continue
        splits = layer.name.split("*")
        layername = splits[0]
        strength = 1.
        if len(splits) == 2:
            strength = float(splits[1])
        splits = [x.strip() for x in layername.split("|")]
        pos, neg = splits if len(splits) == 2 else (splits[0], "")
        layermatrix = torch.zeros(*layers.size)
        _layermatrix = torch.tensor(np.asarray(layer.topil().getchannel("A"))).float() / 255
        layermatrix[layer.top:layer.bottom, layer.left:layer.right] = _layermatrix
        layermatrix = (layermatrix > 0.5).float()

        assert layermatrix.size() == (512, 512)
        unpacked_layers.append({
            "pos": pos,
            "neg": neg,
            "strength": strength,
            tuple(layermatrix.shape): layermatrix,
        })

    fullreskey = tuple(layermatrix.shape)

    # subtract masks from each other before downsampling to reproduce ediffi conditions
    subacc = torch.zeros_like(layermatrix).bool()
    for layer in unpacked_layers[::-1]:
        newlayermask = layer[fullreskey] > 0.5
        newlayermask = newlayermask & (~subacc)
        subacc = subacc | newlayermask
        layer[fullreskey] = newlayermask.to(layer[fullreskey].dtype)

    # compute downsampled versions of the layer masks
    downsamples = [8, 16, 32, 64]
    for layer in unpacked_layers[::-1]:
        layermatrix = layer[fullreskey]
        for downsample in downsamples:
            downsampled = _img_importance_flatten(layermatrix, downsample, downsample)
            layer[tuple(downsampled.shape)] = downsampled

    return {"layers": unpacked_layers, "global": global_descriptions}


class CustomControlLDM(ControlLDM):
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        
        # attach progress to cond["c_crossattn"]
        cond["c_crossattn"].progress = 1 - t / self.num_timesteps
        cond["c_crossattn"].sigma_t = self.sigmas[t]

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, control=None, only_mid_control=self.only_mid_control)
        else:
            if not isinstance(cond["c_crossattn"], SepMixTextConditioning):
                control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond)
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                if self.global_average_pooling:
                    control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, control=control, only_mid_control=self.only_mid_control)
            else:
                cond["c_crossattn"].use_local_cond()
                control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond)
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                if self.global_average_pooling:
                    control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
                local_eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, control=control, only_mid_control=self.only_mid_control)
                
                cond["c_crossattn"].use_global_cond()
                
                control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond)
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                if self.global_average_pooling:
                    control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
                global_eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, control=control, only_mid_control=self.only_mid_control)
                
                eps = cond["c_crossattn"].mix_eps(local_eps, global_eps)
        return eps


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def create_tools():
    model_name = 'control_v11p_sd15_seg'
    model = create_model(f'./models/{model_name}.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
    model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    # cast text encoder to our own
    model.cond_stage_model.__class__ = CustomCLIPTextEmbedder


    model.__class__ = CustomControlLDM
    model.sigmas = ((1 - model.alphas_cumprod) / model.alphas_cumprod) ** 0.5
    model.sigmas = torch.cat([torch.zeros_like(model.sigmas[0:1]), model.sigmas], 0)

    for _module in model.model.diffusion_model.modules():
        if _module.__class__.__name__ == "CrossAttention":
            _module.__class__.forward = custom_forward
            
    return model, ddim_sampler


def compute_cross_attn_masks(spec, device, base_strength=0.):
    layermasks = {}
    for layer in spec["layers"]:
        for k in layer:
            if isinstance(k, tuple):
                if k not in layermasks:
                    layermasks[k] = []
                layermasks[k].append(layer[k] * layer["strength"])

    for k in layermasks:
        # Add global mask; this mask does not interact with others
        layermasks[k] = [torch.ones_like(layermasks[k][0]) * base_strength] + layermasks[k]
        layermasks[k] = torch.stack(layermasks[k], 0).to(device)

    return layermasks


def custom_forward(self, x, context=None, mask=None):
    h = self.heads
    
    if context is not None:
        context = context["c_crossattn"]
        contextembs = context.embs
    else:
        contextembs = x

    q = self.to_q(x)
    k = self.to_k(contextembs)
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
        sim = context.cross_attention_control(sim)
    
    # attention
    sim = (sim * self.scale).softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)

        
def get_cond(prompt, spec, num_samples, method="Ediffi", pww_strength=1.0, pww_threshold=0., pww_softness=0.2, model=None):
    xcond = model.get_learned_conditioning(prompt)
    _cross_attention_masks = compute_cross_attn_masks(spec, model.device)
    cross_attention_masks = {res: torch.index_select(_cross_attention_masks[res], 0, xcond.layer_ids[0]) for res in
                            _cross_attention_masks}
    cross_attention_masks = {res[0] * res[1]: mask.view(mask.size(0), -1).transpose(0, 1)[None] for res, mask in cross_attention_masks.items() if res[0] <= 64}
    xcond.embs = xcond.embs.repeat(num_samples, 1, 1)
    xcond.cross_attn_masks = cross_attention_masks
    
    xcond.set_method(method, customkw=1)
    xcond.strength = pww_strength
    xcond.threshold = pww_threshold
    xcond.softness = pww_softness
    return xcond
    # return {"emb": text_embeddings, "masks": cross_attention_masks}


def process(method, inpfile, segmap, prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, pww_strength, pww_threshold, pww_softness, scale, seed, eta, tools=None):
    # print(f"Loading tools..")
    model, sampler = create_tools() if tools is None else tools
    # print(f"Tools loaded.")
    print(f"PWW strength: {pww_strength}, PWW softness: {pww_softness}")
    
    psdfile = psd_tools.PSDImage.open(inpfile.name)
    spec = unpack_layers(psdfile)        # TODO: add support for extra seeds and sigmas
    
    with torch.no_grad():
        input_image = detected_map = segmap
        
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
            
        prompt, negprompt = prompt.split("|||")
        prompt, negprompt = prompt.strip(), negprompt.strip()
        
        # encode prompts
        x_cond = get_cond(prompt, spec, num_samples, method=method, model=model, pww_strength=pww_strength, pww_threshold=pww_threshold, pww_softness=pww_softness)
        cond = {"c_concat": [control], 
                "c_crossattn": x_cond}
        x_uncond = get_cond(negprompt, spec, num_samples, method="GlobalNeg", model=model, pww_strength=pww_strength, pww_threshold=pww_threshold, pww_softness=pww_softness)
        un_cond = {"c_concat": None if guess_mode else [control], 
                   "c_crossattn": x_uncond}
        
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results


def build_segmap_from_layers(layers):
    ret = None
    for layer in layers:
        randomcolor = torch.rand(3)
        mask = layer[(512,512)]
        maskcolor = mask.unsqueeze(-1).repeat(1, 1, 3) * randomcolor[None, None, :]
        if ret is None:
            ret = torch.zeros_like(maskcolor)
        ret = torch.where(mask.unsqueeze(-1) > 0.5, maskcolor, ret)
        
    ret = (ret * 255).to(torch.long).cpu().numpy()
    return ret


def upload_file(file, aprompt, nprompt):
    file_path = file.name
    # print("file uploaded", file_path)
    layers = psd_tools.PSDImage.open(file_path)
    width, height = layers.size
    init_image = layers.composite()

    spec = unpack_layers(layers)        # TODO: add support for extra seeds and sigmas
    
    segmap = build_segmap_from_layers(spec["layers"])
    global_prompt = spec["global"]["pos"] + ", " + aprompt
    layerprompts = []
    for i, layer in enumerate(spec["layers"]):
        layerprompts.append(layer["pos"] + ", " + aprompt)
        
    prompt = [global_prompt] + layerprompts
    negprompt = [nprompt] * len(prompt)
    prompt = " \n| ".join(prompt)
    prompt = prompt + "\n|||\n" + " \n| ".join(negprompt)
    
    return segmap, prompt


def start_server():
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## PWW with ControlNet and cross-attention control.")
        with gr.Row():
            with gr.Column():
                # input_image = gr.Image(source='upload', type="numpy")
                run_button = gr.Button(label="Run")
                method = gr.Radio(choices=["Ediffi", "Ediffi++", "PosAttn", "PosAttn:BOS", 
                                           "Separate", "SepSwitch", "SepMix"], type="value", value="Ediffi", label="Cross Attention Control")
                pww_strength = gr.Slider(label="PWW Strength", minimum=0.0, maximum=16.0, value=1.0, step=0.25)
                pww_threshold = gr.Slider(label="PWW Threshold", minimum=0.0, maximum=16.0, value=1.0, step=0.25)
                pww_softness = gr.Slider(label="PWW Softness", minimum=0.0, maximum=1.0, value=0.2, step=0.05)
                inpfile = gr.File(file_types=[".psd"], file_count="single")
                # upload_button = gr.UploadButton("Click to Upload a File", file_types=[".psd"], file_count="single")
                # upload_button.upload(upload_file, upload_button, inpfile)
                image_preview = gr.Image(source="canvas", interactive=False)
                # inpfile.change(on_file_change, inpfile, image_preview)
                prompt = gr.Textbox(label="Prompt")
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
                with gr.Accordion("Advanced options", open=False):
                    image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                    strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=40, step=1)
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                    a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                    n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        inpfile.upload(upload_file, [inpfile, a_prompt, n_prompt], [image_preview, prompt])
        run_button.click(fn=process, 
                        inputs=[method, inpfile, image_preview, prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, pww_strength, pww_threshold, pww_softness, scale, seed, eta], 
                        outputs=[result_gallery])
        
    # TODO: use cross-attention control on main net or control net or both?
    
    block.launch(server_name='0.0.0.0')
    
    
def run_default(inpfile="/USERSPACE/lukovdg1/datasets2/datasets/images/balls.psd",
                method="Ediffi",
                pww_strength=1.,
                pww_threshold=0.,
                pww_softness=0.2,
                num_samples=1,
                seed=-1,
                image_resolution=512,
                strength=1.0,
                ddim_steps=40,
                scale=9.0,
                eta=1.0,
                tools=None,
                ):
    guess_mode = False
    a_prompt = 'best quality'
    n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
    class InpFile:
        def __init__(self, f):
            self.name = f
    inpfile = InpFile(inpfile)
    image_preview, prompt = upload_file(inpfile, a_prompt, n_prompt)
    # print(prompt)
    outputs = process(method, inpfile, image_preview, prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, pww_strength, pww_threshold, pww_softness, scale, seed, eta, tools=tools)
    return outputs  # (batsize, 3, H, W)


def run_experiments_ediffi(controlonly=False, controlledonly=False):
    model, sampler = create_tools()
    images = []
    savepath = os.path.join("pww_outputs", "balls_ediffi.pkl")
    seeds = [42, 420, 426, 123, 68, 79, 1337, 234, 1234, 876]
    N = 5

    filepath = "/USERSPACE/lukovdg1/datasets2/datasets/images/balls.psd"
    pww_strengths = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6]
    for c in pww_strengths:
        print(f"pww strength={c}")
        outputs = []
        for seed in seeds[:N]:
            image = run_default(filepath, pww_strength=c, seed=seed, method="Ediffi", 
                                tools=(model, sampler))[0]
            image = Image.fromarray(image)
            outputs.append({"image": image, "seed": seed})
        
        for o in outputs:
            o.update({"inputfile": filepath,
                    "strength": c,
                    "method": "ediffi",})
            images.append(o)

        print("saving to pickle")
        with open(savepath, "wb") as f:
            pickle.dump(images, f)
            
            
def run_experiments_ediffi_pp(controlonly=False, controlledonly=False):
    model, sampler = create_tools()
    images = []
    savepath = os.path.join("pww_outputs", "balls_ediffi_pp.pkl")
    seeds = [42, 420, 426, 123, 68, 79, 1337, 234, 1234, 876]
    N = 5

    filepath = "/USERSPACE/lukovdg1/datasets2/datasets/images/balls.psd"
    pww_strengths = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 2]
    for c in pww_strengths:
        print(f"pww strength={c}")
        outputs = []
        for seed in seeds[:N]:
            image = run_default(filepath, pww_strength=c, seed=seed, method="Ediffi++",
                                tools=(model, sampler))[0]
            image = Image.fromarray(image)
            outputs.append({"image": image, "seed": seed})
        
        for o in outputs:
            o.update({"inputfile": filepath,
                    "strength": c,
                    "method": "ediffi_pp",})
            images.append(o)

        print("saving to pickle")
        with open(savepath, "wb") as f:
            pickle.dump(images, f)
            
            
def run_experiments_posattn(controlonly=False, controlledonly=False):
    model, sampler = create_tools()
    images = []
    savepath = os.path.join("pww_outputs", "balls_posattn.pkl")
    seeds = [42, 420, 426, 123, 68, 79, 1337, 234, 1234, 876]
    N = 5
    s = 0.2

    filepath = "/USERSPACE/lukovdg1/datasets2/datasets/images/balls.psd"
    pww_thresholds = [0.3, 0.5, 0.7, 0.2, 0.1, 0.]
    for t in pww_thresholds:
        pww_strengths = [1., 2., 3., 4.]
        if t <= 0.5:
            pww_strengths.append(5.)
        if t <= 0.3:
            pww_strengths.append(10.)
        if t == 0.:
            pww_strengths = [0.]
        for c in pww_strengths:
            print(f"pww threshold={t}, strength={c}, softness={s}")
            outputs = []
            for seed in seeds[:N]:
                image = run_default(filepath, pww_strength=c, pww_threshold=t, pww_softness=s, seed=seed, method="PosAttn",
                                    tools=(model, sampler))[0]
                image = Image.fromarray(image)
                outputs.append({"image": image, "seed": seed})
            
            for o in outputs:
                o.update({"inputfile": filepath,
                        "strength": c,
                        "threshold": t,
                        "softness": s,
                        "method": "posattn",})
                images.append(o)

            print("saving to pickle")
            with open(savepath, "wb") as f:
                pickle.dump(images, f)
                

def run_experiments_posattn_with_bos(controlonly=False, controlledonly=False):
    model, sampler = create_tools()
    images = []
    savepath = os.path.join("pww_outputs", "balls_posattn_with_bos.pkl")
    seeds = [42, 420, 426, 123, 68, 79, 1337, 234, 1234, 876]
    N = 5
    s = 0.2

    filepath = "/USERSPACE/lukovdg1/datasets2/datasets/images/balls.psd"
    pww_thresholds = [-1., 0.25, 0.5, 0.75, 1.5]
    for t in pww_thresholds:
        pww_strengths = [1., 2., 5., 10]
        if t == -1:
            pww_strengths = [0]
        for c in pww_strengths:
            print(f"pww threshold={t}, strength={c}, softness={s}")
            outputs = []
            for seed in seeds[:N]:
                image = run_default(filepath, pww_strength=c, pww_threshold=t, pww_softness=s, seed=seed, method="PosAttn:WithBOS",
                                    tools=(model, sampler))[0]
                image = Image.fromarray(image)
                outputs.append({"image": image, "seed": seed})
            
            for o in outputs:
                o.update({"inputfile": filepath,
                        "strength": c,
                        "threshold": t,
                        "softness": s,
                        "method": "posattn",})
                images.append(o)

            print("saving to pickle")
            with open(savepath, "wb") as f:
                pickle.dump(images, f)
                

def run_experiments_posattn_raise_bos(controlonly=False, controlledonly=False):
    model, sampler = create_tools()
    images = []
    savepath = os.path.join("pww_outputs", "balls_posattn_raise_bos.pkl")
    seeds = [42, 420, 426, 123, 68, 79, 1337, 234, 1234, 876]
    N = 5
    s = 0.2

    filepath = "/USERSPACE/lukovdg1/datasets2/datasets/images/balls.psd"
    pww_thresholds = [0.3, 0.5, 0.7, 0.2, 0.1, 0.]
    for t in pww_thresholds:
        pww_strengths = [1., 2., 3., 4.]
        if t <= 0.5:
            pww_strengths.append(5.)
        if t <= 0.3:
            pww_strengths = [10.] + pww_strengths
        if t == 0.:
            pww_strengths = [0.]
        for c in pww_strengths:
            print(f"pww threshold={t}, strength={c}, softness={s}")
            outputs = []
            for seed in seeds[:N]:
                image = run_default(filepath, pww_strength=c, pww_threshold=t, pww_softness=s, seed=seed, 
                                    method="PosAttn:RaiseBOS", tools=(model, sampler))[0]
                image = Image.fromarray(image)
                outputs.append({"image": image, "seed": seed})
            
            for o in outputs:
                o.update({"inputfile": filepath,
                        "strength": c,
                        "threshold": t,
                        "softness": s,
                        "method": "posattn:bos",})
                images.append(o)

            print("saving to pickle")
            with open(savepath, "wb") as f:
                pickle.dump(images, f)
                

def run_experiments_separate(controlonly=False, controlledonly=False):
    model, sampler = create_tools()
    images = []
    savepath = os.path.join("pww_outputs", "balls_separate.pkl")
    seeds = [42, 420, 426, 123, 68, 79, 1337, 234, 1234, 876]
    N = 5
    s = 0.2

    filepath = "/USERSPACE/lukovdg1/datasets2/datasets/images/balls.psd"
    pww_thresholds = [0.1, 0.25, 2., 0.5, 0.75]  #[0.3, 0.5, 0.7, 0.2, 0.1, 0.]
    for t in pww_thresholds:
        pww_strengths = [10., 1., 4.]
        for c in pww_strengths:
            print(f"pww threshold={t}, strength={c}, softness={s}")
            outputs = []
            for seed in seeds[:N]:
                image = run_default(filepath, pww_strength=c, pww_threshold=t, pww_softness=s, seed=seed, 
                                    method="Separate", tools=(model, sampler))[0]
                image = Image.fromarray(image)
                outputs.append({"image": image, "seed": seed})
            
            for o in outputs:
                o.update({"inputfile": filepath,
                        "strength": c,
                        "threshold": t,
                        "softness": s,
                        "method": "posattn:separate",})
                images.append(o)

            print("saving to pickle")
            with open(savepath, "wb") as f:
                pickle.dump(images, f)
                
                
def run_experiments_sepswitch(controlonly=False, controlledonly=False):
    model, sampler = create_tools()
    images = []
    savepath = os.path.join("pww_outputs", "balls_sepswitch_vsoft.pkl")
    seeds = [42, 420, 426, 123, 68, 79, 1337, 234, 1234, 876]
    N = 5
    s = 0.5

    filepath = "/USERSPACE/lukovdg1/datasets2/datasets/images/balls.psd"
    pww_thresholds = [0., 00.05, 0.075, 0.1, 0.25]  #[0.3, 0.5, 0.7, 0.2, 0.1, 0.]
    for s in [0.1, 0.]:
        for t in pww_thresholds:
            pww_strengths = [1., 1.5, 2., 4., 10]
            for c in pww_strengths:
                print(f"pww threshold={t}, strength={c}, softness={s}")
                outputs = []
                for seed in seeds[:N]:
                    image = run_default(filepath, pww_strength=c, pww_threshold=t, pww_softness=s, seed=seed, 
                                        method="SepSwitch", tools=(model, sampler))[0]
                    image = Image.fromarray(image)
                    outputs.append({"image": image, "seed": seed})
                
                for o in outputs:
                    o.update({"inputfile": filepath,
                            "strength": c,
                            "threshold": t,
                            "softness": s,
                            "method": "posattn:sepswitch",})
                    images.append(o)

                print("saving to pickle")
                with open(savepath, "wb") as f:
                    pickle.dump(images, f)
                
                
def run_experiments_sepmix(controlonly=False, controlledonly=False):
    model, sampler = create_tools()
    images = []
    savepath = os.path.join("pww_outputs", "balls_sepmix.pkl")
    seeds = [42, 420, 426, 123, 68, 79, 1337, 234, 1234, 876]
    N = 5
    s = 0.2

    filepath = "/USERSPACE/lukovdg1/datasets2/datasets/images/balls.psd"
    pww_thresholds = [0.05, 0.1, 0.25, 0.5]  #[0.3, 0.5, 0.7, 0.2, 0.1, 0.]
    for t in pww_thresholds:
        pww_strengths = [1., 0.25, 0.5, 0.75]
        for c in pww_strengths:
            print(f"pww threshold={t}, strength={c}, softness={s}")
            outputs = []
            for seed in seeds[:N]:
                image = run_default(filepath, pww_strength=c, pww_threshold=t, pww_softness=s, seed=seed, 
                                    method="SepMix", tools=(model, sampler))[0]
                image = Image.fromarray(image)
                outputs.append({"image": image, "seed": seed})
            
            for o in outputs:
                o.update({"inputfile": filepath,
                        "strength": c,
                        "threshold": t,
                        "softness": s,
                        "method": "posattn:sepmix",})
                images.append(o)

            print("saving to pickle")
            with open(savepath, "wb") as f:
                pickle.dump(images, f)
                
                
def main(server=False,
         ediffi=False,
         ediffipp=False,
         posattn=False,
         posattnwbos=False,
         posattnbos=False,
         separate=False,
         sepswitch=False,
         sepmix=False,
         controlonly=False,
         controlledonly=False,
         ):
    assert sum([server, ediffi, ediffipp, posattn, posattnbos, separate]) <= 1
    if server: start_server()
    elif ediffi: print("Running ediffi experiments") ; run_experiments_ediffi(controlonly=controlonly, controlledonly=controlledonly)
    elif ediffipp: print("Running ediffi experiments") ; run_experiments_ediffi_pp(controlonly=controlonly, controlledonly=controlledonly)
    elif posattn: run_experiments_posattn(controlonly=controlonly, controlledonly=controlledonly)
    elif posattnwbos: run_experiments_posattn_with_bos(controlonly=controlonly, controlledonly=controlledonly)
    elif posattnbos: run_experiments_posattn_raise_bos(controlonly=controlonly, controlledonly=controlledonly)
    elif separate: run_experiments_separate(controlonly=controlonly, controlledonly=controlledonly)
    elif sepswitch: run_experiments_sepswitch(controlonly=controlonly, controlledonly=controlledonly)
    elif sepmix: run_experiments_sepmix(controlonly=controlonly, controlledonly=controlledonly)
    #else: print("Choose what to run") ; run_experiments_sepmix(controlonly=controlonly, controlledonly=controlledonly)
    else: print("Choose what to run") ; run_experiments_sepswitch(controlonly=controlonly, controlledonly=controlledonly)
    #else: print("Choose what to run") ; run_experiments_posattn_with_bos(controlonly=controlonly, controlledonly=controlledonly)
                            

if __name__ == "__main__":
    fire.Fire(main)
    
    # TODO: implement SepMix with decaying Posattn and decaying BOS
    # TODO: what happens if we disable cross attention control on controlling model or on controlled model?
    # TODO: verify ediffi does not include BOS token
    
