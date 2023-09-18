
from copy import deepcopy
import json
from pathlib import Path
import pickle as pkl
import fire
import os
from PIL import Image
import numpy as np

import torch
from cldm.logger import ImageLogger, nested_to

from dataset import COCODataLoader, COCOPanopticDataset
from ldm.util import SeedSwitch, seed_everything
from train_controlnet_pww import create_controlnet_pww_model
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
    expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_posattn2_v4.1_exp_21",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_posattn_v4.1_exp_1",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_posattn_v4.1_exp_3",
        #  expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_global_v4.1_exp_2/",
        #   expdir="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_bothext2_v4.1_exp_2_forreal/",
         loadckpt="latest*.ckpt",
         numgen=4,
         examples="extradev.examples.pkl",
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
        
    # load dev set from pickle
    loadedexamples = []
    for devexamplepath in examples.split(","):
        with open(devexamplepath, "rb") as f:
            loadedexamples_e = pkl.load(f)
            loadedexamples += loadedexamples_e
            
    # override pickled defaults
    valid_ds = COCOPanopticDataset(examples=loadedexamples, casmode=cas + "+test", simpleencode=simpleencode, 
                                   mergeregions=mergeregions, limitpadding=limitpadding, 
                                   max_masks=28 if limitpadding else 10)
    # valid_dl = COCODataLoader(valid_ds, batch_size=numgen, num_workers=1, shuffle=False, repeatexample=True)
    
    # which ckpt to load
    loadckpt = list(expdir.glob(loadckpt))
    assert len(loadckpt) in (0, 1)
    if len(loadckpt) == 1:
        loadckpt = loadckpt[0]
        args["loadckpt"] = str(loadckpt)
    else:
        loadckpt = ""
        print("ckpt not found, not loading")
    
    model = create_controlnet_pww_model(cas_name=cas, freezedown=freezedown, simpleencode=simpleencode, 
                                        threshold=threshold, strength=strength, softness=softness, 
                                        loadckpt=loadckpt)
    model.limitpadding = args["limitpadding"]
    
    print("model loaded")
    
    imagelogger = ImageLogger(batch_frequency=999, dl=None, seed=seed)
    
    i = 1
    exppath = expdir / f"generated_{i}"
    while exppath.exists():
        i += 1
        exppath = expdir / f"generated_{i}"
        
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