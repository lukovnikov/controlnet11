
from copy import deepcopy
import json
from pathlib import Path
import pickle as pkl
import fire
import os
from PIL import Image
import numpy as np
from einops import rearrange, repeat

import torch
from torchvision.utils import make_grid
from cldm.logger import ImageLogger, nested_to

from dataset import COCODataLoader, COCOPanopticDataset
from ldm.util import SeedSwitch, seed_everything
from train_controlnet_pww import CASMode, ControlPWWLDM, create_controlnet_pww_model
import torchvision


def tensor_to_pil(x, rescale=True):
    if rescale:
        x = (x + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
    x = x.numpy()
    x = (x * 255).astype(np.uint8)
    return Image.fromarray(x)


def write_image_grid(savedir, images, i, batsize, rescale=True):
    for k in images:
        grid = torchvision.utils.make_grid(images[k], nrow=batsize)
        grid = tensor_to_pil(grid)
        filename = f"{i}.png"
        path = savedir / filename
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        grid.save(path)
        
        
def do_log_img(imagelogger, batch, pl_module):
    is_train = pl_module.training
    if is_train:
        pl_module.eval()
        
    device = pl_module.device
    numgenerated = 0
    
    batch = nested_to(batch, device)

    with torch.no_grad():
        images = pl_module.log_images_reencode(batch, split="train", **imagelogger.log_images_kwargs)
        
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


class ControlPWWLDM_DDIMInvert(ControlPWWLDM):
    @torch.no_grad()
    def log_images_reencode(self, batch, N=None, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True, partial_steps=35,
                   **kwargs):
        use_ddim = True

        log = dict()
        N = batch["image"].shape[0]
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        # N = min(z.shape[0], N)
        log["reconstruction"] = reconstrimg = self.decode_first_stage(z)  #.clamp(0, 1) * 2.0 - 1.0
        log["control"] = controlimg = c_cat * 2.0 - 1.0
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        # if plot_diffusion_rows:
        #     n_row = min(z.shape[0], n_row)
        #     # get diffusion row
        #     diffusion_row = list()
        #     z_start = z[:n_row]
        #     for t in range(self.num_timesteps):
        #         if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
        #             t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
        #             t = t.to(self.device).long()
        #             noise = torch.randn_like(z_start)
        #             z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        #             diffusion_row.append(self.decode_first_stage(z_noisy))

        #     diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
        #     diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
        #     diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
        #     diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
        #     log["diffusion_row"] = diffusion_grid

        # if sample:
        #     # get denoise row
        #     samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
        #                                              batch_size=N, ddim=use_ddim,
        #                                              ddim_steps=ddim_steps, eta=ddim_eta)
        #     x_samples = self.decode_first_stage(samples)
        #     log["samples"] = x_samples
        #     if plot_denoise_rows:
        #         denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
        #         log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uncond_batch = self.get_uncond_batch(batch)
            _, uc = self.get_input(uncond_batch, self.first_stage_key, bs=N)
            uc_cat, uc_cross = uc["c_concat"][0], uc["c_crossattn"][0]
            
            # generate images
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning={"c_concat": [uc_cat], "c_crossattn": [uc_cross]},
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = generated_img = x_samples_cfg
            
            # reverse DDIM
            samples_reversed_xT, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning={"c_concat": [uc_cat], "c_crossattn": [uc_cross]},
                                             reverse=True, x0=samples_cfg, partial_steps=partial_steps,
                                             )
            samples_reconstructed, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning={"c_concat": [uc_cat], "c_crossattn": [uc_cross]},
                                             x_T=samples_reversed_xT, partial_steps=partial_steps,
                                             )
            reconstructed_img = self.decode_first_stage(samples_reconstructed)
            
        log[f"all"] = torch.cat([reconstrimg, controlimg, generated_img, reconstructed_img], 2)
        del log["reconstruction"]
        del log["control"]
        del log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"]

        return log


def main(
    expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.2/checkpoints_coco_global_v4.2_exp_1",
    # expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.2/checkpoints_coco_legacy-NewEdiffipp_v4.2_exp_4",
    # expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.2/checkpoints_coco_cac_v4.2_exp_1",
    # expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.2/checkpoints_coco_dd_v4.2_exp_4",
    # expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.2/checkpoints_coco_posattn4_v4.2_exp_3",
    # expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.2/checkpoints_coco_posattn5_v4.2_exp_3",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_posattn_v4.1_exp_1",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_posattn_v4.1_exp_3",
        #  expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_global_v4.1_exp_2/",
        #   expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_bothext2_v4.1_exp_2_forreal/",
         loadckpt="latest*.ckpt",
         numgen=5,
         examples="extradev.examples.pkl",
        #  examples="threeballs1.pkl",
        #  examples="threefruits1.pkl",
        #  examples="foursquares1.pkl",
        #  examples="evaldata/openair1.pkl",
        # examples="extradev.examples.pkl,evaldata/threeballs1.pkl,evaldata/threefruits1.pkl,evaldata/foursquares1.pkl,evaldata/openair1.pkl",
        #  examples="threeballs.variants.pkl", #"coco2017.4dev.examples.pkl,extradev.examples.pkl", # "extradev.examples.pkl",  #"coco2017.4dev.examples.pkl",
         devices=(3,),
         seed=123456,
         threshold=-1.,
         softness=-1.,
         strength=-1.,
         limitpadding=False,
         ):    
    localargs = locals().copy()
    expdir = Path(expdir)
    with open(expdir / "args.json") as f:
        trainargs = json.load(f)
        
    args = deepcopy(trainargs)
    for k, v in localargs.items():
        if v == -1.:
            pass
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
        args["loadckpt"] = str(loadckpt)
    else:
        loadckpt = ""
        print("ckpt not found, not loading")
    
    model = create_controlnet_pww_model(casmode=cas, freezedown=freezedown, simpleencode=simpleencode, 
                                        threshold=threshold, strength=strength, softness=softness, 
                                        loadckpt=loadckpt)
    model.limitpadding = args["limitpadding"]
    model.__class__ = ControlPWWLDM_DDIMInvert
    
    print("model loaded")
        
    # load dev set from pickle
    for devexamplepath in examples.split(","):
        with open(devexamplepath, "rb") as f:
            loadedexamples = pkl.load(f)
            
        # override pickled defaults
        valid_ds = COCOPanopticDataset(examples=loadedexamples, casmode=cas +"test", simpleencode=simpleencode, 
                                    mergeregions=mergeregions, limitpadding=limitpadding, 
                                    max_masks=28 if limitpadding else 10)
        # valid_dl = COCODataLoader(valid_ds, batch_size=numgen, num_workers=1, shuffle=False, repeatexample=True)
        
        imagelogger = ImageLogger(batch_frequency=999, dl=None, seed=seed)
        
        i = 1
        exppath = expdir / f"generated_{Path(devexamplepath).name}_{i}"
        while exppath.exists():
            i += 1
            exppath = expdir / f"generated_{Path(devexamplepath).name}_{i}"
            
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
            _examples = [valid_ds.materialize_example(example) for _ in range(numgen)]
            _batch = valid_ds.collate_fn(_examples)
            images = do_log_img(imagelogger, _batch, model)
            write_image_grid(exppath, images, i, batsize=numgen, rescale=imagelogger.rescale)
            outputexamples.append([])
            for image in images["all"]:
                src_img, seg_img, out_img = image.chunk(3, 1)
                outexample = deepcopy(example)
                outexample.image_data = tensor_to_pil(out_img)
                outexample.seg_data2 = tensor_to_pil(seg_img)
                outputexamples[-1].append(outexample)
        
        with open(exppath / "outbatches.pkl", "wb") as f:
            pkl.dump(outputexamples, f)
            
        print(f"saved to file")
            
        
    
    
if __name__ == "__main__":
    fire.Fire(main)