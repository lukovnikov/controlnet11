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
        
    def cross_attention_control(self, sim):
        """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
        sim = sim + self.weight_func(sim)
        return sim
    
    def weight_func(self, sim):
        return self._weight_func(self, sim)
    
    def set_method(self, methodstr, **kw):
        method_functions = {
            "Ediffi": ediffi_weight_func,
            "Ediffi++": ediffi_pp_weight_func,
            "RaiseAll": posattn_weight_func,
            "RaiseBOS": posattn_raise_bos_weight_func,
        }
        if methodstr in method_functions:
            f = method_functions[methodstr]
            f = partial(f, **kw)
            self._weight_func = f
        else:
            raise Exception(f"unknown method: {methodstr}")
        

def ediffi_weight_func(self, sim, sigma_square=False, qk_std=False, **kw):
    pass    # TODO
    ret = strength * w * math.log(
                1 + (sigma ** 2 if sigma_square else sigma)) * (qk.std() if qk_std else qk.max())
    return sim


def ediffi_pp_weight_func(self, sim, sigma_square=True, qk_std=True, **kw):
    return ediffi_weight_func(self, sim, sigma_square=sigma_square, qk_std=qk_std, **kw)


def _threshold_f(p, a, b=1): # transitions from 0 at p=a to 1 at p=b using sinusoid curve
    threshold = (a, b)
    a = min(threshold)
    b = max(threshold)
    if p < a:
        weight = 0
    elif p >= a and p < b:
        midpoint = (b - a) / 2 + a
        weight = (math.sin(math.pi * (p - midpoint) / (b - a)) + 1) * 0.5
    else:
        weight = 1
    return weight


def posattn_weight_func(self, sim, **kw):
    pass    # TODO
    a, b = threshold, 1
    if isinstance(threshold, tuple) and len(threshold) == 2:
        a = min(threshold)
        b = max(threshold)
    if isinstance(threshold_bos, tuple) and threshold_bos[0] == None:
        threshold_bos = (1, 1)
    if isinstance(threshold_eos, tuple) and threshold_eos[0] == None:
        threshold_eos = (1, 1)
        
    weight = 1 - _threshold_f(progress, a, b)

    # compute cross-attention mask added weight for BOS and EOS tokens
    bos_weight = _threshold_f(progress, *threshold_bos)
    eos_weight = _threshold_f(progress, *threshold_eos)
    # bos_weight = min(bos_weight, weight)    # BOS added weight is at most the base added weight
    # eos_weight = min(eos_weight, weight)    # EOS added weight is at most the base added weight

    bos_w = torch.zeros_like(w)
    bos_w[:, 0] = 1 * bos_weight

    eos_w = torch.zeros_like(w)
    if eos_position is not None:
        eos_w[:, eos_position] = 1 * eos_weight

    w = w + bos_w + eos_w

    wprime = w[None] * mult * weight * (qk.max() if not use_std else qk.std(-1, keepdim=True))
    return wprime
    return sim


def posattn_raise_bos_weight_func(self, sim, **kw):
    pass    # TODO
    return sim
        
        
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
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            if self.global_average_pooling:
                control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, control=control, only_mid_control=self.only_mid_control)

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


        
        
def get_cond(prompt, spec, num_samples, method="Ediffi", model=None):
    xcond = model.get_learned_conditioning(prompt)
    _cross_attention_masks = compute_cross_attn_masks(spec, model.device)
    cross_attention_masks = {res: torch.index_select(_cross_attention_masks[res], 0, xcond.layer_ids[0]) for res in
                            _cross_attention_masks}
    cross_attention_masks = {res[0] * res[1]: mask.view(mask.size(0), -1).transpose(0, 1)[None] for res, mask in cross_attention_masks.items() if res[0] <= 64}
    xcond.embs = xcond.embs.repeat(num_samples, 1, 1)
    xcond.cross_attn_masks = cross_attention_masks
    
    xcond.set_method(method, customkw=1)
    return xcond
    # return {"emb": text_embeddings, "masks": cross_attention_masks}


def process(method, inpfile, segmap, prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    print(f"Loading tools..")
    model, sampler = create_tools()
    print(f"Tools loaded.")
    
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
        x_cond = get_cond(prompt, spec, num_samples, method=method, model=model)
        cond = {"c_concat": [control], 
                "c_crossattn": x_cond}
        x_uncond = get_cond(negprompt, spec, num_samples, method=method, model=model)
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
    print("file uploaded", file_path)
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
                method = gr.Radio(choices=["Ediffi", "Ediffi++", "RaiseAll", "RaiseBOS"], type="value", value="Ediffi", label="Cross Attention Control")
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
                    detect_resolution = gr.Slider(label="Preprocessor Resolution", minimum=128, maximum=1024, value=512, step=1)
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=40, step=1)
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                    a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                    n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        inpfile.upload(upload_file, [inpfile, a_prompt, n_prompt], [image_preview, prompt])
        run_button.click(fn=process, 
                        inputs=[method, inpfile, image_preview, prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta], 
                        outputs=[result_gallery])
        
    # TODO: use cross-attention control on main net or control net or both?
    
    block.launch(server_name='0.0.0.0')
    
    
def run_default(inpfile="/USERSPACE/lukovdg1/datasets2/datasets/images/balls.psd"):
    method = "Ediffi"
    image_preview = gr.Image(source="canvas", interactive=False)
    prompt = gr.Textbox(label="Prompt")
    num_samples = 1
    seed = -1
    image_resolution = 512
    strength = 1.0
    guess_mode = False
    detect_resolution = 512
    ddim_steps = 40
    scale = 9.0
    eta = 1.0
    a_prompt = 'best quality'
    n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
    class InpFile:
        def __init__(self, f):
            self.name = f
    inpfile = InpFile(inpfile)
    image_preview, prompt = upload_file(inpfile, a_prompt, n_prompt)
    outputs = process(method, inpfile, image_preview, prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
                            

if __name__ == "__main__":
    run_default()
    # start_server()
    
