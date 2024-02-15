import math
from copy import deepcopy
import json
from pathlib import Path
import pickle as pkl
import fire
import os
from PIL import Image, ImageFilter
import numpy as np
import random

import torch
from cldm.logger import ImageLogger, nested_to

from dataset import COCODataLoader, COCOPanopticDataset, to_tensor, randomcolor_hsv, _tokenize_annotated_prompt, _img_importance_flatten
from ldm.util import SeedSwitch, seed_everything
from train_controlnet_pww import CASMode, create_controlnet_pww_model
import torchvision

from blurlib import gaussian_blur, hardblur, variable_hardblur, variable_softblur


class COCOPanopticDatasetTransformed(COCOPanopticDataset):
    def cond_transform(self, cond_imgtensor, bgr_color=None, soft_blur=None, hard_blur=None, var_soft_blur=None, var_hard_blur=None):
        soft_blur = soft_blur if soft_blur is not None else self.soft_blur
        hard_blur = hard_blur if hard_blur is not None else self.hard_blur
        var_soft_blur = var_soft_blur if var_soft_blur is not None else self.var_soft_blur
        var_hard_blur = var_hard_blur if var_hard_blur is not None else self.var_hard_blur
        if var_hard_blur > 0:
            cond_imgtensor = variable_hardblur(cond_imgtensor, rmap=var_hard_blur, bgr_color=bgr_color)
        if hard_blur > 0:
            cond_imgtensor = hardblur(cond_imgtensor, radius=hard_blur, bgr_color=bgr_color)
            
        if var_soft_blur > 0:
            cond_imgtensor = variable_softblur(cond_imgtensor, rmap=var_soft_blur, bgr_color=bgr_color)
        if soft_blur > 0:
            cond_imgtensor = gaussian_blur(cond_imgtensor, kernel_size=soft_blur)
        return cond_imgtensor
    
    def materialize_example(self, example, soft_blur=None, hard_blur=None, var_soft_blur=None, var_hard_blur=None):
        def region_code_to_rgb(rcode):
            B = rcode // 256**2
            rcode = rcode % 256**2
            G = rcode // 256
            R = rcode % 256
            return (R, G, B)
            
        # materialize one example
        # 1. load image and segmentation map
        img = example.load_image()   #Image.open(self.image_db[example_id]["path"]).convert("RGB")
        seg_img = example.load_seg_image()   #Image.open(self.panoptic_db[example_id]["segments_map"]).convert("RGB")
        
        if self.upscale_to is not None:
            upscalefactor = self.upscale_to / min(img.size)
            newsize = [int(s * upscalefactor) for s in img.size]
            img = img.resize(newsize, resample=Image.BILINEAR)
            seg_img = seg_img.resize(newsize, resample=Image.BOX)
            
        # 2. transform to tensors
        imgtensor = to_tensor(img)
        seg_imgtensor = torch.tensor(np.array(seg_img)).permute(2, 0, 1)
        
        # 3. create conditioning image by randomly swapping out colors
        
        bgr_color = torch.tensor(randomcolor_hsv())
        cond_imgtensor = torch.ones_like(imgtensor) * bgr_color[:, None, None]
        
        # 4. pick one caption at random (TODO: or generate one from regions)
        captions = [random.choice(example.captions)]
    
        # 4. load masks
        masks = [torch.ones_like(imgtensor[0], dtype=torch.bool)]
        # get the captions of the regions and build layer ids
        # region_code_to_layerid = {0: 0}
        region_code_to_layerid = {}
        
        region_caption_to_layerid = {}
        unique_region_captions = set()
        
        for _, region_info in example.seg_info.items():
            unique_region_captions.add(region_info["caption"])
            
        if not (self.regiondrop is False):
            all_unique_region_captions = list(unique_region_captions)
            random.shuffle(all_unique_region_captions)
            unique_region_captions = all_unique_region_captions[:self.max_masks]
            if isinstance(self.regiondrop, float):
                assert 0. <= self.regiondrop <= 1.
                unique_region_captions = [c for c in unique_region_captions if random.random() > self.regiondrop]
                if len(unique_region_captions) < self.min_masks:
                    unique_region_captions = all_unique_region_captions[:self.min_masks]
            unique_region_captions = set(unique_region_captions)
        
        for i, (region_code, region_info) in enumerate(example.seg_info.items()):
            rgb = torch.tensor(region_code_to_rgb(region_code))
            region_mask = (seg_imgtensor == rgb[:, None, None]).all(0)
            if (region_mask > 0).sum() / np.prod(region_mask.shape) < self.min_region_area:
                continue
            if self.casmode is None or self.casmode.name != "global":
                region_caption = region_info["caption"]
                if region_caption in unique_region_captions:
                    if (not self.mergeregions) or (region_caption not in region_caption_to_layerid):
                        new_layerid = len(masks)
                        region_caption_to_layerid[region_caption] = new_layerid
                        captions.append(region_info["caption"])
                        masks.append(region_mask)
                    else:
                        new_layerid = region_caption_to_layerid[region_caption]
                        masks[new_layerid] = masks[new_layerid] | region_mask        
                    region_code_to_layerid[region_code] = new_layerid            
                else:
                    pass #continue    # or pass? (if pass, the mask will end up in the conditioning image for controlnet)
            
            randomcolor = torch.tensor(randomcolor_hsv()) #if self.casmode is not None else torch.tensor(randomcolor_predef())
            maskcolor = region_mask.unsqueeze(0).repeat(3, 1, 1) * randomcolor[:, None, None]
        
            cond_imgtensor = torch.where(region_mask.unsqueeze(0) > 0.5, maskcolor, cond_imgtensor)
            
        original_cond_imgtensor = cond_imgtensor
        cond_imgtensor = self.cond_transform(cond_imgtensor, 
                                             bgr_color=bgr_color, 
                                             soft_blur=soft_blur, 
                                             hard_blur=hard_blur,
                                             var_soft_blur=var_soft_blur,
                                             var_hard_blur=var_hard_blur)
        
        # append extra global prompt
        extraexpressions = ["This image contains", "In this image are", "In this picture are", "This picture contains"]
        # if (self.casmode.name == "doublecross") and not ("keepprompt" in self.casmode.chunks or "test" in self.casmode.chunks):
        #     # assert not self.simpleencode
        #     captions[0] += ". " + random.choice(extraexpressions) + " " + ", ".join([e for e in captions[1:]]) + "."
            
            
        minimize_length = False
        
        if self.casmode.use_global_prompt_only:   # self.casmode in ("posattn-opt", "posattn2-opt"):
            caption = captions[0] if not self.casmode.augment_only else ""
            if self.casmode.augment_global_caption:       # if training, automatically augment sentences
                tojoin = []
                for i, capt in enumerate(captions[1:]):
                    tojoin.append(f"{{{capt}:{i}}}")
                caption += " " + random.choice(extraexpressions) + " " + ", ".join(tojoin) + "."
            _caption, _layerids = _tokenize_annotated_prompt(caption, tokenizer=self.tokenizer, minimize_length=minimize_length)
            # replace region codes with layer ids
            for region_code, layerid in region_code_to_layerid.items():
                _layerids = torch.masked_fill(_layerids, _layerids == region_code + 1, layerid)
            captions, layerids = [_caption], [_layerids]
            encoder_layerids = [torch.ones_like(layerids[-1]) * 0]
        # elif self.simpleencode: #  or (self.casmode in ("posattn-opt", "posattn2-opt") and not ("keepprompt" in self.casmodechunks or "test" in self.casmodechunks)):   # or self.casmode == "doublecross":
        #     tojoin = []
        #     for i, capt in enumerate(captions[1:]):
        #         tojoin.append(f"{{{capt}:{i}}}")
        #     captions[0] += ". " + random.choice(extraexpressions) + " " + ", ".join(tojoin) + "."
            
        #     _captions, _layerids = _tokenize_annotated_prompt(captions[0], tokenizer=self.tokenizer, minimize_length=minimize_length)
        #     captions, layerids = [_captions], [_layerids]
        #     encoder_layerids = [torch.ones_like(layerids[-1]) * 0]
        else:
            # encode separately
            tokenized_captions = []
            layerids = []
            encoder_layerids = []
             #self.casmode != "global"
             
            if self.casmode.augment_global_caption:
                caption = captions[0] if not self.casmode.augment_only else ""
                caption += ". " + random.choice(extraexpressions) + " " + ", ".join([e for e in captions[1:]]) + "."
                captions[0] = caption
             
            for i, caption in enumerate(captions):
                if i == 0:
                    tokenized_caption, layerids_e = _tokenize_annotated_prompt(caption, tokenizer=self.tokenizer, minimize_length=minimize_length)
                    # replace region codes with layer ids
                    for region_code, layerid in region_code_to_layerid.items():
                        layerids_e = torch.masked_fill(layerids_e, layerids_e == region_code + 1, layerid)
                    tokenized_captions.append(tokenized_caption)
                    layerids.append(layerids_e)    
                else:
                    tokenized_captions.append(self.tokenize(caption, tokenizer=self.tokenizer, minimize_length=minimize_length)[0])
                    layerids.append(torch.ones_like(tokenized_captions[-1]) * i)    
                encoder_layerids.append(torch.ones_like(layerids[-1]) * i)
            captions = tokenized_captions
            if self.casmode.replace_layerids_with_encoder_layerids:
                layerids[0] = encoder_layerids[0]
                
        if self.limitpadding:
            for i in range(len(captions)):
                caption = captions[i]
                numberpads = (caption == self.tokenizer.pad_token_id).sum()     # assumption: all pads are contiguous and at the end
                endidx = -(numberpads - self.padlimit)
                captions[i] = caption[:endidx]
                layerids[i] = layerids[i][:endidx]
                encoder_layerids[i] = encoder_layerids[i][:endidx]

        # random square crop of size divisble by 64 and maximum size 512
        cropsize = min((min(imgtensor[0].shape) // 64) * 64, 512)
        crop = (random.randint(0, imgtensor.shape[1] - cropsize), 
                random.randint(0, imgtensor.shape[2] - cropsize))
        # print(cropsize)
        
        imgtensor = imgtensor[:, crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize]
        cond_imgtensor = cond_imgtensor[:, crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize]
        original_cond_imgtensor = original_cond_imgtensor[:, crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize]
        masks = [maske[crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize] for maske in masks]
        
        # compute downsampled versions of the layer masks
        downsamples = [cropsize // e for e in [8, 16, 32, 64]]
        downmaskses = []
        for mask in masks:
            downmasks = {}
            for downsample in downsamples:
                downsampled = _img_importance_flatten(mask.float(), downsample, downsample)
                downmasks[tuple(downsampled.shape)] = downsampled
            downmaskses.append(downmasks)
            
        # concatenate masks in one tensor
        downmasktensors = {}
        for downmasks in downmaskses:
            for res, downmask in downmasks.items():
                if res not in downmasktensors:
                    downmasktensors[res] = []
                downmasktensors[res].append(downmask)
        downmasktensors = {k: torch.stack(v, 0) for k, v in downmasktensors.items()}
        
        # DONE: provide conditioning image based on layers
        
        imgtensor = imgtensor * 2 - 1.
        
        return ({"image": imgtensor, 
                "cond_image": cond_imgtensor,
                "captions": captions,
                "layerids": layerids,
                "encoder_layerids": encoder_layerids,
                "regionmasks": downmasktensors
                },
                {"image": imgtensor, 
                "cond_image": original_cond_imgtensor,
                "captions": captions,
                "layerids": layerids,
                "encoder_layerids": encoder_layerids,
                "regionmasks": downmasktensors
                })


def tensor_to_pil(x, rescale=True):
    if rescale:
        x = (x + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
    x = x.numpy()
    x = (x * 255).astype(np.uint8)
    return Image.fromarray(x)


def write_image_grid(savedir, images, i, batsize, rescale=True, suffix=""):
    for k in images:
        grid = torchvision.utils.make_grid(images[k], nrow=batsize)
        grid = tensor_to_pil(grid)
        filename = f"{i}{suffix}.png"
        path = savedir / filename
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        grid.save(path)
        
        
def do_log_img(imagelogger, batch, pl_module, seed=None):
    if seed is not None:
        seed_everything(seed)
        
    is_train = pl_module.training
    if is_train:
        pl_module.eval()
        
    device = pl_module.device
    numgenerated = 0
    
    batch = nested_to(batch, device)

    with torch.no_grad():
        images = pl_module.log_images(batch, split="train", **imagelogger.log_images_kwargs)
        
    for k in images:
        N = images[k].shape[0]
        numgenerated += N
        images[k] = images[k][:N]
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().cpu()
            if imagelogger.clamp:
                images[k] = torch.clamp(images[k], -1., 1.)

    if is_train:
        pl_module.train()
        
    return images


def main(
         expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints_vp/v2/checkpoints_coco_global_v2_exp_1",
         loadckpt="latest*.ckpt",
         numgen=5,
         examples="evaldata_vp/clearshapes.pkl",
         devices=(2,),
         seed=123456,
         threshold=-1.,
         softness=-1.,
         strength=-1.,
         limitpadding=False,
         softblur=0,           # 31, 61, 101
         hardblur=0,          # 20
         varsoftblur=61,
         varhardblur=0,
         ):    
    print(torch.__version__)
    localargs = locals().copy()
    expdir = Path(expdir)
    with open(expdir / "args.json") as f:
        trainargs = json.load(f)
        
    args = deepcopy(trainargs)
    for k, v in localargs.items():      # override original args with args specified here
        if v == -1.:
            pass
        elif k == "loadckpt":
            args[k] = args[k]
        else:
            args[k] = v
    
    # unpack args
    cas = args["cas"]
    simpleencode = args["simpleencode"]
    mergeregions = args["mergeregions"]
    limitpadding = args["limitpadding"]
    freezedown = args["freezedown"]
    threshold = args["threshold"]
    softness = args["softness"]
    strength = args["strength"]
    
    seed_everything(seed)
    
    # print(args)
    print(json.dumps(args, indent=4))     
    print(devices, type(devices), devices[0])
    
    cas = CASMode(cas)
    
    # which ckpt to load
    loadckpt = list(expdir.glob(loadckpt))
    assert len(loadckpt) in (0, 1)
    if len(loadckpt) == 1:
        loadckpt = loadckpt[0]
        args["loadckpt"] += "," + str(loadckpt)
    elif len(loadckpt) > 1:
        raise Exception("multiple matches for loadckpt, unclear")
    else:
        print("ckpt not found, not loading")
    
    model = create_controlnet_pww_model(casmode=cas, freezedown=freezedown, simpleencode=simpleencode, 
                                        threshold=threshold, strength=strength, softness=softness, 
                                        loadckpt=args["loadckpt"])
    model.limitpadding = args["limitpadding"]
    
    print("model loaded")
        
    # load dev set from pickle
    for devexamplepath in examples.split(","):
        with open(devexamplepath, "rb") as f:
            loadedexamples = pkl.load(f)
            
        # override pickled defaults
        valid_ds = COCOPanopticDatasetTransformed(examples=loadedexamples, casmode=cas + "test", simpleencode=simpleencode, 
                                    mergeregions=mergeregions, limitpadding=limitpadding, 
                                    max_masks=28 if limitpadding else 10)
        # valid_dl = COCODataLoader(valid_ds, batch_size=numgen, num_workers=1, shuffle=False, repeatexample=True)
        
        imagelogger = ImageLogger(batch_frequency=999, dl=None, seed=seed)
        
        i = 1
        extraspec =   ("" if softblur == 0 else f"_softblur_r{softblur}") \
                    + ("" if hardblur == 0 else f"_hardblur_r{hardblur}")\
                    + ("" if varsoftblur == 0 else f"_varsoftblur_r{varsoftblur}")\
                    + ("" if varhardblur == 0 else f"_varhardblur_r{varhardblur}")
        exppath = expdir / f"generated_{Path(devexamplepath).name}{extraspec}_{i}"
        while exppath.exists():
            i += 1
            exppath = expdir / f"generated_{Path(devexamplepath).name}{extraspec}_{i}"
            
        exppath.mkdir(parents=True, exist_ok=False)
        print(f"logging in {exppath}")
        with open(exppath / "args.json", "w") as f:
            json.dump(args, f, indent=4)
        
        device = torch.device("cuda", devices[0])
        print("generation device", device)
        model = model.to(device)
        
        allexamples = []
        for _, v in valid_ds.examples:
            for example in v:
                allexamples.append(example)
                
        print("total examples:", len(allexamples))
        outputexamples = []
        for i, example in enumerate(allexamples):
            _examples, _examples_original = zip(*[valid_ds.materialize_example(example, 
                                                                               soft_blur=softblur, 
                                                                               hard_blur=hardblur,
                                                                               var_soft_blur=varsoftblur,
                                                                               var_hard_blur=varhardblur) for _ in range(numgen)])
            _batch, _batch_original = valid_ds.collate_fn(_examples), valid_ds.collate_fn(_examples_original)
            seed = random.randint(10000, 100000)
            images = do_log_img(imagelogger, _batch, model, seed=seed)
            write_image_grid(exppath, images, i, batsize=numgen, rescale=imagelogger.rescale)
            images2 = do_log_img(imagelogger, _batch_original, model, seed=seed)
            write_image_grid(exppath, images2, i, batsize=numgen, rescale=imagelogger.rescale, suffix="_original")
            outputexamples.append([])
            for image, image2 in zip(images["all"], images2["all"]):
                src_img, seg_img, out_img = image.chunk(3, 1)
                src_img2, seg_img2, out_img2 = image2.chunk(3, 1)
                outexample = deepcopy(example)
                outexample.image_data = tensor_to_pil(out_img)
                outexample.image_data_original = tensor_to_pil(out_img2)
                outexample.seg_data_src = outexample.seg_data
                outexample.seg_data = tensor_to_pil(seg_img)
                outexample.seg_data_original = tensor_to_pil(seg_img2)
                outputexamples[-1].append(outexample)
        
        with open(exppath / "outbatches.pkl", "wb") as f:
            pkl.dump(outputexamples, f)
            
        print(f"saved to file")
            
        
    
    
if __name__ == "__main__":
    fire.Fire(main)