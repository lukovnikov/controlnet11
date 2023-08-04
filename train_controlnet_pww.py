from copy import deepcopy
from datetime import timedelta
import json
import pickle as pkl
from pathlib import Path
import random
from typing import Any, Dict
import cv2
import fire
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
from dataset import COCOPanopticDataset, COCODataLoader
from ldm.modules.attention import BasicTransformerBlock, default
from ldm.modules.diffusionmodules.util import torch_cat_nested
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import SeedSwitch, log_txt_as_img

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
        
    # def cross_attention_control(self, sim, numheads=1):
    #     """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
    #     wf = self.weight_func(sim)
        
    #     wf = wf[:, None].repeat(1, numheads, 1, 1)
    #     wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
    #     sim = sim + wf
    #     return sim
    
    # def weight_func(self, sim):
    #     mask = self.cross_attn_masks[sim.shape[1]].to(sim.dtype)
    #     ret = mask * sim.std() * self.strength
    #     return ret
    
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
    
    
class CustomCrossAttentionBase(nn.Module):
    
    @classmethod
    def from_base(cls, m):
        m.__class__ = cls
        m.init_extra()
        return m
    
    def init_extra(self):
        for p in self.get_trainable_parameters():
            p.train_param = True
    
    def weight_func(self, sim, context, sim_mask=None):
        with torch.no_grad():
            if sim_mask is None:
                sim_mask = context.captiontypes >= 0
                sim_mask = repeat(sim_mask, 'b j -> (b h) () j', h=sim.shape[0] // sim_mask.shape[0])
            simstd = torch.masked_select(sim, sim_mask).std()
            mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)
            ret = mask * simstd * context.strength
        return ret
    
    def get_trainable_parameters(self):
        return []    

    
class CustomCrossAttentionBaseline(CustomCrossAttentionBase):
    # Tries to emulate the basic setting where only the global prompt is available, and attention computation is not changed.

    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        if context is not None:
            context = context["c_crossattn"][0]
            contextembs = context.embs
        else:
            assert False        # this shouldn't be used as self-attention
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
            sim = self.cross_attention_control(sim, context, numheads=h)
        
        # attention
        sim = (sim * self.scale).softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 0
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
        wf = self.weight_func(sim, context, sim_mask=mask)
        wf.masked_fill_(~context.global_prompt_mask[:, None, :], max_neg_value)
        
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        sim.masked_fill_(~mask, max_neg_value)
        sim = sim + wf
        return sim
    
    def get_trainable_parameters(self):
        params = list(self.to_q.parameters())
        params += list(self.to_k.parameters())
        return params
    

class CustomCrossAttentionBaselineBoth(CustomCrossAttentionBaseline):
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 0
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        sim.masked_fill_(~mask, max_neg_value)
        """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
        wf = self.weight_func(sim, context)
        
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        sim = sim + wf
        return sim
    

class CustomCrossAttentionBaselineLocal(CustomCrossAttentionBaseline):
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 2
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
        wf = self.weight_func(sim, context, sim_mask=mask)
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        wf.masked_fill_(~mask, max_neg_value)
        sim = sim + wf
        return sim
    
    
class CustomCrossAttentionBaselineLocalGlobalFallback(CustomCrossAttentionBaseline):
    """ Uses only local descriptions, unless there is none (all local ones are masked), then falls back to global description."""
    
    def cross_attention_control(self, sim, context, numheads=None):
        # compute mask that ignores everything except the local descriptions
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 2
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        
        # get mask that selects global as well as applicable local prompt
        wf = self.weight_func(sim, context, sim_mask=mask)

        # expand dimensions to match heads
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        # remove the global part from the mask
        wf.masked_fill_(~mask, 0)  # max_neg_value)
        # determine where to use global mask (where no local descriptions are available)
        useglobal = wf.sum(-1) == 0
        
        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        wfmaxes = wf.max(-1, keepdim=True)[0]
        wf = (wf >= (wfmaxes.clamp_min(1e-3) - 1e-4)).float()
        
        # get mask that selects only global tokens
        gmask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        gmask = repeat(gmask, 'b j -> (b h) () j', h=numheads)
        
        # update stimulation to attend to global tokens when no local tokens are available
        wf = wf + (useglobal[:, :, None] & gmask)
        
        sim.masked_fill_(wf == 0, max_neg_value)
        # sim = sim + wf
        return sim
    

class CustomCrossAttentionSepSwitch(CustomCrossAttentionBaseline):
    threshold = 0.2
    """ Uses only local descriptions, unless there is none (all local ones are masked), then falls back to global description."""
    
    def cross_attention_control(self, sim, context, numheads=None):
        # compute mask that ignores everything except the local descriptions
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 2
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        
        # get mask that selects global as well as applicable local prompt
        wf = self.weight_func(sim, context, sim_mask=mask)

        # expand dimensions to match heads
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        # remove the global part from the mask
        wf.masked_fill_(~mask, 0)  # max_neg_value)
        # determine where to use global mask (where no local descriptions are available)
        useglobal = wf.sum(-1) == 0
        
        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        wfmaxes = wf.max(-1, keepdim=True)[0]
        wf = (wf >= (wfmaxes.clamp_min(1e-3) - 1e-4)).float()
        
        # get mask that selects only global tokens
        gmask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        gmask = repeat(gmask, 'b j -> (b h) () j', h=numheads)
        
        # update stimulation to attend to global tokens when no local tokens are available
        lmask = wf + (useglobal[:, :, None] & gmask)
        
        prog = context.progress
        prog = prog[:, None].repeat(1, self.heads).view(-1)
        lorg = prog <= self.threshold
        
        mask = torch.where(lorg[:, None, None], lmask, gmask)
        
        sim.masked_fill_(mask == 0, max_neg_value)
        # sim = sim + wf
        return sim
    
    
class CustomCrossAttentionBaselineGlobal(CustomCrossAttentionBaseline):
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        sim.masked_fill_(~mask, max_neg_value)
        return sim
    
    
class DoublecrossBasicTransformerBlock(BasicTransformerBlock):
    @classmethod
    def convert(cls, m):
        m.__class__ = cls
        m.init_extra()
        return m
    
    def init_extra(self):
        self.attn2l = deepcopy(self.attn2)
        self.attn2.__class__ = CustomCrossAttentionBaselineGlobal
        self.attn2l.__class__ = CustomCrossAttentionBaselineLocalGlobalFallback
        
        self.register_buffer("manual_gate", torch.tensor([1.]))
        self.learned_gate = torch.nn.Parameter(torch.tensor([0.]))      # TODO: per-head learned gate
        
        self.norm2l = deepcopy(self.norm2)
        
        for p in self.get_trainable_parameters():
            p.train_param = True
        
    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.manual_gate * torch.tanh(self.learned_gate) * self.attn2l(self.norm2l(x), context=context) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    def get_trainable_parameters(self):
        params = list(self.attn2l.parameters())
        params += list(self.norm2l.parameters())
        params += [self.learned_gate]
        return params   
    
    
# class DoubleCrossAttention(CustomCrossAttentionBaselineGlobal):         # TODO remove this
#     """ First applies trainable cross-attention using local descriptions and then regular frozen global attention"""
    
#     @classmethod
#     def from_base(cls, m):
#         m.__class__ = cls
#         m.init_extra()
#         return m
    
#     def init_extra(self):
#         self.local = deepcopy(self)     # this will be the trainable one
#         self.local.__class__ = CustomCrossAttentionBaselineLocalGlobalFallback
        
#         self.register_buffer("manual_gate", torch.tensor([1.]))
#         self.learned_gate = torch.nn.Parameter(torch.tensor([0.]))
        
#         for p in self.get_trainable_parameters():
#             p.train_param = True
        
#     def forward(self, x, context=None, mask=None):
#         # 1. apply (trainable) local attention first
#         localout = self.local(self.norm1(x), context=context, mask=mask)
#         x = x + self.manual_gate * self.learned_gate * localout
#         # 2. apply self (non-trainable) global-only attention
#         globalout = super().forward(self.norm2(x), context=context, mask=mask)
#         x = x + globalout
#         return globalout 
    
#     def get_trainable_parameters(self):
#         params = list(self.local.parameters())
#         return params   

        
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
    
    
class CustomCrossAttentionExt(CustomCrossAttentionBase):
    # DONE: add model extension to be able to tell where is global and local parts of the prompt
        
    def init_extra(self):
        # conditioning on token type (global BOS, global or local)
        self.token_type_emb = TokenTypeEmbedding(self.to_k.in_features)
        # conditioning on progress (0..1)
        self.progress_emb = ProgressEmbedding(self.to_q.in_features)
        
        for p in self.get_trainable_parameters():
            p.train_param = True

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
            sim = self.cross_attention_control(sim, context, numheads=h)
        
        # attention
        sim = (sim * self.scale).softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 0
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        sim.masked_fill_(~mask, max_neg_value)
        """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
        wf = self.weight_func(sim, context)
        
        wf = wf[:, None].repeat(1, numheads, 1, 1)      # TODO: rewrite with einops
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        sim = sim + wf
        return sim
    
    def get_trainable_parameters(self):
        params = list(self.to_q.parameters())
        params += list(self.to_k.parameters())
        params += list(self.token_type_emb.parameters())
        params += list(self.progress_emb.parameters())
        return params
    
        
class TokenTypeEmbeddingMinimal(torch.nn.Module):
    def __init__(self, embdim):
        super().__init__()
        self.emb = torch.nn.Embedding(10, embdim)
        
    def forward(self, tokentypes):
        tokentypeemb = self.emb(tokentypes.clamp_min(0))
        return tokentypeemb
    
    
class ProgressEmbeddingMinimal(torch.nn.Module):
    def __init__(self, embdim) -> None:
        super().__init__()
        self.progress_emb = torch.nn.Sequential(
            torch.nn.Linear(1, embdim),
            torch.nn.ReLU(),
            torch.nn.Linear(embdim, embdim)
        )
        # self.gateA = torch.nn.Parameter(torch.randn(embdim) * 1e-3)
        
    def forward(self, progress):
        progressemb = self.progress_emb(progress)
        return progressemb
    
    
class DiscretizedProgressEmbed(torch.nn.Module):
    def __init__(self, embdim, embeds=50, steps=1000) -> None:
        super().__init__()
        self.embdim, self.embeds, self.steps = embdim, embeds, steps
        self.emb1 = torch.nn.Embedding(embeds+1, embdim)
        self.emb2 = torch.nn.Embedding(steps // embeds, embdim)
        
    def forward(self, x):
        # if torch.any(x == 1.):
        #     print("x contains a 1")
        xstep = (x * self.steps).round().long().clamp_max(self.steps-1)
        x1 = torch.div(xstep, (self.steps // self.embeds) , rounding_mode="floor")
        x2 = xstep % (self.steps // self.embeds)
        emb1 = self.emb1(x1)
        emb2 = self.emb2(x2)
        return emb1 + emb2
    
    
class ProgressClassifier(torch.nn.Module):      # classifies whether to use global prompt or local prompt for every head given progress
    INITBIAS = -3
    def __init__(self, embdim=512, numheads=8) -> None:
        super().__init__()
        self.embdim, self.numheads, self.numclasses = embdim, numheads, 2
        self.net = torch.nn.Sequential(
            DiscretizedProgressEmbed(embdim),
            torch.nn.GELU(),
            torch.nn.Linear(embdim, embdim),
            torch.nn.GELU(),
            torch.nn.Linear(embdim, self.numclasses * numheads)
        )
        finalbias = self.net[-1].bias
        classbias = torch.tensor([0, self.INITBIAS]).repeat(finalbias.shape[0]//2)
        finalbias.data += classbias
        
    def forward(self, progress):
        out = self.net(progress)        # maps (batsize, 1) to (batsize, numclases * numheads)
        probs = out.view(out.shape[0], self.numheads, self.numclasses).softmax(-1)
        return probs
    
    
class CustomCrossAttentionMinimal(CustomCrossAttentionExt):
    # Minimal cross attention: computes scores based on content independently from scores based on progress and region
    
    def init_extra(self):
        self.progressclassifier = ProgressClassifier(numheads=self.heads)
        # # conditioning on token type (global BOS, global or local)
        # self.token_type_emb = TokenTypeEmbeddingMinimal(self.to_k.out_features)
        # # conditioning on progress (0..1)
        # self.progress_emb = ProgressEmbeddingMinimal(self.to_q.out_features)
        # # gate parameters: one trainable scalar for every head in this attention layer
        # self.gate = torch.nn.Parameter(torch.randn(self.heads) * 1e-6)      # 
        
        for p in self.get_trainable_parameters():
            p.train_param = True
        
    def get_trainable_parameters(self):
        params = list(self.progressclassifier.parameters())
        # params = list(self.token_type_emb.parameters())
        # params += list(self.progress_emb.parameters())
        # params += [self.gate]
        return params
        
    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        if context is not None:
            context = context["c_crossattn"][0]
            contextembs = context.embs
        else:
            assert False        # this shouldn't be used as self-attention
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
            sim = self.cross_attention_control(sim, context, numheads=h)
        else:
            sim = (sim * self.scale).softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 2
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        
        # get mask that selects global as well as applicable local prompt
        wf = self.weight_func(sim, context, sim_mask=mask)
        wfscale = wf.max()

        # expand dimensions to match heads
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        # remove the global part from the mask
        wf.masked_fill_(~mask, 0)  # max_neg_value)
        # determine where to use global mask (where no local descriptions are available)
        useglobal = wf.sum(-1) == 0
        
        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        wfmaxes = wf.max(-1, keepdim=True)[0]
        wf = (wf >= (wfmaxes.clamp_min(1e-3) - 1e-4))
        
        # get mask that selects only global tokens
        gmask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        gmask = repeat(gmask, 'b j -> (b h) () j', h=numheads)
        
        # update stimulation to attend to global tokens when no local tokens are available
        lmask = wf | (useglobal[:, :, None] & gmask)
        
        gsim = sim.masked_fill(gmask==0, max_neg_value)
        gattn = (gsim * self.scale).softmax(dim=-1)
        
        lsim = sim.masked_fill(lmask==0, max_neg_value)
        lattn = (lsim * self.scale).softmax(dim=-1)
        
        progressclasses = self.progressclassifier(context.progress).view(-1, 2)
        attn = gattn.float() * progressclasses[:, 0][:, None, None] + lattn.float() * progressclasses[:, 1][:, None, None]
        return attn
        
        
        # cas_mask = gmask.float() * progressclasses[:, 0][:, None, None] + lmask.float() * progressclasses[:, 1][:, None, None]
        # cas_mask = cas_mask * wfscale
        # sim = sim + cas_mask
        
        # sim.masked_fill_((lmask | gmask) == 0, max_neg_value)
        # sim = sim + wf
        # sim = (sim * self.scale).softmax(dim=-1)
        # return sim


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
        # params = []
        # # select query and key projections as well as new modules from CustomCrossAttention
        # for param in self.parameters():
        #     if hasattr(param, "train_param") and param.train_param:
        #         params += param
        # for module in self.modules():
        #     if isinstance(module, (DoublecrossBasicTransformerBlock, CustomCrossAttentionBase)):
        #         params += list(module.get_trainable_parameters())
                
        # for param in params:
        #     param.store_param = True
        
        params = []
        saved_param_names = []
        for paramname, param in self.named_parameters():
            if hasattr(param, "train_param") and param.train_param:
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
        
        # new_caption = [[] for _ in range(batch["caption"].shape[0])]
        # new_layerids = [[] for _ in new_caption]
        # new_captiontypes = [[] for _ in new_caption]
        
        new_caption2 = torch.ones_like(batch["caption"]) * eos
        new_layerids2 = torch.ones_like(batch["layerids"]) * -1
        new_captiontypes2 = torch.ones_like(batch["captiontypes"]) * -1
        
        new_regionmasks = {k: torch.zeros_like(v) for k, v in batch["regionmasks"].items()}
        # device = batch["caption"].device
        caption = batch["caption"].cpu()
        
        # layerids = batch["layerids"].cpu()
        # captiontypes = batch["captiontypes"].cpu()
        
        prev = None
        for i in range(len(caption)):
            k = 0
            for j in range(len(caption[0])):
                cur = caption[i, j].item()
                if cur == bos or (cur == eos and prev != eos):
                    # new_caption[i].append(cur)
                    # new_layerids[i].append(layerids[i, j].item())
                    # new_captiontypes[i].append(captiontypes[i, j].item())
                    new_caption2[i, k] = batch["caption"][i, j]
                    new_layerids2[i, k] = batch["layerids"][i, j]
                    new_captiontypes2[i, k] = batch["captiontypes"][i, j]
                    for res in new_regionmasks:
                        new_regionmasks[res][i, k] = batch["regionmasks"][res][i, j]
                    k += 1
                prev = cur
                
        # maxlen = caption.shape[1]
        # for i in range(len(new_caption)):
        #     while len(new_caption[i]) < maxlen:
        #     # for j in range(len(new_caption[i]), maxlen):
        #         new_caption[i].append(eos)
        #         new_layerids[i].append(-1)
        #         new_captiontypes[i].append(-1)
                
        uncond_cond["caption"] = new_caption2  #torch.tensor(new_caption).to(device)
        uncond_cond["layerids"] = new_layerids2  #torch.tensor(new_layerids).to(device)
        uncond_cond["captiontypes"] = new_captiontypes2  #torch.tensor(new_captiontypes).to(device)
        uncond_cond["regionmasks"] = new_regionmasks
                
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
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = generated_img = x_samples_cfg
            
        log[f"all"] = torch.cat([reconstrimg, controlimg, generated_img], 2)
        del log["reconstruction"]
        del log["control"]
        del log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"]

        return log
    

class ControlPWWLDMSimpleEncode(ControlPWWLDM):
    
    def get_learned_conditioning(self, cond):   return self.get_learned_conditioning_simple(cond)
    
    def get_learned_conditioning_simple(self, cond):
        # takes conditioning info (cond_key) and preprocesses it to later be fed into LDM
        # returns CustomTextConditioning object
        # called from get_input()
        # must be used with cond_key = "all", then get_input() passes the batch as-is in here
        # DONE: unpack texts, embed them, pack back up and package with cross-attention masks
        
        # this is a non-parallelized implementation
        with torch.no_grad():
            tokenids = cond["caption"]
            layerids = cond["layerids"]
             
            # 2. encode using text encoder
            outputs = self.cond_stage_model.transformer(input_ids=tokenids, output_hidden_states=self.cond_stage_model.layer=="hidden")
            if self.cond_stage_model.layer == "last":
                text_emb = outputs.last_hidden_state
            elif self.cond_stage_model.layer == "pooled":
                text_emb = outputs.pooler_output[:, None, :]
            else:
                text_emb = outputs.hidden_states[self.layer_idx]
            
        global_prompt_mask = cond["captiontypes"] < 2
        global_bos_eos_mask = cond["captiontypes"] == 0
        
        ret = CustomTextConditioning(embs=text_emb,
                                     layer_ids=layerids,
                                     token_ids=tokenids,
                                     global_prompt_mask=global_prompt_mask,
                                     global_bos_eos_mask=global_bos_eos_mask)
        
        ret.captiontypes = cond["captiontypes"]
        
        cross_attn_masks = cond["regionmasks"]    
        cross_attn_masks = {res[0] * res[1]: mask.view(mask.size(0), mask.size(1), -1).transpose(1, 2) for res, mask in cross_attn_masks.items() if res[0] <= 64}
        ret.cross_attn_masks = cross_attn_masks
        return ret
    
    
def convert_model(model, cas_class=None, cas_name=None, freezedown=False, simpleencode=False, threshold=-1):
    model.__class__ = ControlPWWLDM
    if simpleencode:
        model.__class__ = ControlPWWLDMSimpleEncode
    model.first_stage_key = "image"
    model.control_key = "cond_image"
    model.cond_stage_key = "all"
    
    if cas_name is not None:
        assert cas_class is None
        cas_class = {"both": CustomCrossAttentionBaselineBoth,
                     "local": CustomCrossAttentionBaselineLocal,
                     "global": CustomCrossAttentionBaselineGlobal,
                     "bothext": CustomCrossAttentionExt,
                     "bothminimal": CustomCrossAttentionMinimal,
                     "doublecross": None,
                     "sepswitch": CustomCrossAttentionSepSwitch,
                     }[cas_name]
        
    if cas_class is None:
        cas_class = CustomCrossAttentionBaseline
        
    print(f"CAS name: {cas_name}")
    print(f"CAS class: {cas_class}")
    
    # DONE: replace CrossAttentions that are at attn2 in BasicTransformerBlocks with adapted CustomCrossAttention that takes into account cross-attention masks
    for module in model.model.diffusion_model.modules():
        if isinstance(module, BasicTransformerBlock): # module.__class__.__name__ == "BasicTransformerBlock":
            assert not module.disable_self_attn
            if cas_name == "doublecross":
                DoublecrossBasicTransformerBlock.convert(module)
            else:
                module.attn2 = cas_class.from_base(module.attn2)
                module.attn2.threshold = threshold
        
            
    for module in model.control_model.modules():
        if isinstance(module, BasicTransformerBlock): # module.__class__.__name__ == "BasicTransformerBlock":
            assert not module.disable_self_attn
            if cas_name == "doublecross":
                DoublecrossBasicTransformerBlock.convert(module)
            else:
                module.attn2 = cas_class.from_base(module.attn2)
                module.attn2.threshold = threshold
    
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


def create_controlnet_pww_model(basemodelname="v1-5-pruned.ckpt", model_name='control_v11p_sd15_seg', cas_name="bothext",
                                freezedown=False, simpleencode=False, threshold=-1, loadckpt=""):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(f'./models/{model_name}.yaml').cpu()
    # load main weights
    model.load_state_dict(load_state_dict(f'./models/{basemodelname}', location='cpu'), strict=False)
    # load controlnet weights
    model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cpu'), strict=False)
    model.base_model_name = basemodelname
    model.controlnet_model_name = model_name
    
    model = convert_model(model, cas_name=cas_name, freezedown=freezedown, simpleencode=simpleencode, threshold=threshold)
    
    if loadckpt != "":
        refparam1a = model.model.diffusion_model.middle_block[1].proj_in.weight.data.clone()
        refparam2a = deepcopy(model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2l.to_q.weight.data.clone())
        ckpt_state_dict = load_state_dict(loadckpt, location="cpu")
        # testing the partial loading
        model.load_state_dict(ckpt_state_dict, strict=False)
        refparam1b = model.model.diffusion_model.middle_block[1].proj_in.weight.data.clone()
        refparam2b = deepcopy(model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2l.to_q.weight.data.clone())
        assert torch.all(refparam1a == refparam1b)
    return model


def main(batsize=5,
         version="v3",
         datadir="/USERSPACE/lukovdg1/coco2017/",
         devexamples="coco2017.4dev.examples.pkl",
         cas="doublecross",  # "both", "local", "global", "bothext", "bothminimal", "doublecross", "sepswitch"
         devices=(0,),
         numtrain=-1,
         forreal=False,
         seed=12345,        # seed for training
         log_image_seed=41,     # seed for generating logging images
         freezedown=False,      # don't adapt the down-sampling blocks of the Unet, only change and train the upsamling blocks
         simpleencode=False,    # encode both global and all local prompts as one sequence
        #  minimal=False,         # only interaction between layer+head info, progress and token type (global/local/bos) (<-- essentially a learned schedule for CAS, independent of content)
         generate="dev",   # ""
         threshold=-1,
         loadckpt="", #"/USERSPACE/lukovdg1/controlnet11/checkpoints/v2/checkpoints_coco_doublecross_v2_exp_1_forreal/interval_delta_epoch=epoch=5_step=step=64069.ckpt", #"",
         ):  
    args = locals().copy()
    # print(args)
    print(json.dumps(args, indent=4))     
    ### usage
    # simpleencode=True: only makes sense with both or bothext
    if simpleencode:
        assert cas in ("bothext", "bothminimal")
    print(devices, type(devices), devices[0])
    # Configs
    batch_size = batsize
    logger_freq = 1000 if forreal else 10 #300
    learning_rate = 1e-5
    sd_locked = False
    
    generate_set = "dev" if generate.lower() == "true" else generate
    generate = generate != ""
    
    numtrain = None if numtrain == -1 else numtrain
    
    expnr = 1
    def get_exp_name(_expnr):
        ret = f"checkpoints/{version}/checkpoints_coco_{cas}_{version}_exp_{_expnr}{'_forreal' if forreal else ''}"
        if freezedown:
            ret += "_freezedown"
        if simpleencode:
            ret += "_simpleencode"
        if cas in ("sepswitch",):
            ret += f"_threshold={threshold}"
        return ret
        
    exppath = Path(get_exp_name(expnr))
    while exppath.exists():
        expnr += 1
        exppath = Path(get_exp_name(expnr))
        
    # load dev set from pickle
    with open(devexamples, "rb") as f:
        loadedexamples = pkl.load(f)
    # override pickled defaults
    valid_ds = COCOPanopticDataset(examples=loadedexamples, casmode=cas, simpleencode=simpleencode)
    valid_dl = COCODataLoader(valid_ds, batch_size=4, num_workers=4, shuffle=False)
    
    model = create_controlnet_pww_model(cas_name=cas, freezedown=freezedown, simpleencode=simpleencode, 
                                        threshold=threshold, loadckpt=loadckpt)
    
    seedswitch = SeedSwitch(seed, log_image_seed)
    image_logger = ImageLogger(batch_frequency=logger_freq, dl=valid_dl, seed=seedswitch)
    
    if generate:
        exppath.mkdir(parents=True, exist_ok=False)
        image_logger.do_log_img(model, split="gen")
    
    else:
        ds = COCOPanopticDataset(maindir=datadir, split="train" if forreal else "valid", casmode=cas, simpleencode=simpleencode, 
                        max_samples=numtrain if numtrain is not None else (None if forreal else 1000))
        
        print(len(ds))
        batsizes = {384: round(batch_size * 2.4), 448:round(batch_size * 1.4), 512: batch_size}
        print(f"Batch sizes: {batsizes}")
        dl = COCODataLoader(ds, batch_size=batsizes, 
                            num_workers=max(batsizes.values()),
                            shuffle=True)

        model.learning_rate = learning_rate if not generate else 0
        model.sd_locked = sd_locked
            
        checkpoints = get_checkpointing_callbacks(interval=8*60*60 if forreal else 60*10, dirpath=exppath)
        logger = pl.loggers.TensorBoardLogger(save_dir=exppath)
        
        max_steps = -1
        if generate:
            max_steps = 1
        
        trainer = pl.Trainer(accelerator="gpu", devices=devices, 
                            precision=32, max_steps=max_steps,
                            logger=logger,
                            callbacks=checkpoints + [image_logger])

        # Train!
        print(f"Writing to {exppath}")
        
        # with open(exppath / "args.json", "w") as f:
        #     json.dump(args, f)
        trainer.fit(model, dl)
        
        # # generate
        # device = torch.device("cuda", devices[0])
        # print("device", device)
        # model = model.to(device)
        # image_logger.log_img(model, None, 0, split="dev")
    
    
if __name__ == "__main__":
    fire.Fire(main)
    
    # DONE: implement conditioning on which prompt type a token is and also on progress in CustomCrossAttention
    # DONE: implement weight function in CustomTextConditioning
    # DONE: select the right parameters to train (all the CustomCrossAttentions at first)
    # DONE: implement generation of images (log_images())
    # DONE: implement checkpointing
    #   DONE: save only the changed parameters
    #   DONE: test checkpointing
    # DONE: how to handle negative prompt? --> replace prompt of all regions and global prompt with ""
    # DONE: check if everything works as expected
    # DONE: check how just global prompt works without any changes (CustomAttentionBaseline)
    
    # DONE: why are images looking washed out? loss intended for -1:1 range has 0:1 range targets?
    # DONE: log the same images throughout the training process
    # WONTDOTODO: create variant where we change only the mid and upsampling blocks <-- probably not necessary
    
    # DONE: create variant where we encode global and local prompts together as one prompt passing through encoder as one sequence (instead of separately)
    # DONE: create minimal variant where there is only interaction between layer+head info, progress and token type (global/local/bos) (<-- essentially a learned schedule for CAS, independent of content)
    # WONTDOFORNOWTODO: create variant that combines regular and simple encode
    
    # TODO: train CAS better by dropping out the ControlNet control (because ControlNet already implies some object identity with its shapes)
    
    # TODO: IDEA: double-cross-attention: one cross attention on global prompt (unchanged, untrained?), and another cross-attention on local prompts
    
    # DONE: validation setup: take images of apples and oranges and of cats and dogs and change where the cats and dogs and apples and oranges are
    # DONE: use panoptic segmentation data from COCO instead
    # TODO(?): drop some regions
    
    # TODO: evaluation setup --> RegionCLIP?
    
    # COCO: Interesting test examples:
    # 152353, 577451, 326390, 265531, 494550  (oranges)      <-- round things
    # 289170, 513424, 415222 (apples)                        <-- round things
    # 303308, 86408, 315645 (microwave)                      <-- boxy things
    # 56886, 285258, 385186, 387876, 268726, 475663 (dog)            <-- animal things
    # 401758, 228764, 460872, 524594            <--- cat and dog
    # 572448, 543192, 25411, 42641      <-- apples and oranges
    
    # BASELINES
    # DONE: implement clean global cas here
    # DONE: implement switching from local-only to global-only
    # TODO: implement global-only prompt with annotation
    
    # TODO: implement loading already trained models
    # TODO: port the rabbitfire and balls examples
    # TODO: implement sampling on entire dev set
    