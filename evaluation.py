from pathlib import Path
from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel
from cldm.logger import nested_to
from dataset import COCOPanopticDataset, COCOPanopticExample, COCODataLoader
import pickle

import numpy as np


def load_clip_model(modelname="openai/clip-vit-base-patch32"):
    model = CLIPModel.from_pretrained(modelname)
    processor = CLIPProcessor.from_pretrained(modelname)
    return model, processor


def display_example(x):
    img = x.load_image()
    seg_img = x.load_seg_image()
    print(x.captions)
    print(repr(x.seg_info))
    return None


def region_code_to_rgb(rcode):
    B = rcode // 256**2
    rcode = rcode % 256**2
    G = rcode // 256
    R = rcode % 256
    return (R, G, B)


def rgb_to_regioncode(r, g, b):
    ret = r + g * 256 + b * (256**2)
    return ret
    
    
def run():
    model, processor = load_clip_model()
    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    print(probs)
    
    
def do_example(x, model, processor, tightcrop=True, fill="none"):
    regionimages = []
    regiondescriptions = []
    
    image = np.array(x.load_image())
    seg_image = np.array(x.load_seg_image())
    seg_image_codes = rgb_to_regioncode(*np.split(seg_image, 3, -1))
    for regioncode, regioninfo in x.seg_info.items():
        mask = (seg_image_codes == regioncode) * 1.
        height, width, _ = mask.shape
        
        if tightcrop:
            bbox_left = np.where(mask > 0)[1].min()
            bbox_right = np.where(mask > 0)[1].max()
            bbox_top = np.where(mask > 0)[0].min()
            bbox_bottom = np.where(mask > 0)[0].max()
            
            bbox_size = ((bbox_right-bbox_left), (bbox_bottom - bbox_top))
            bbox_center = (bbox_left + bbox_size[0] / 2, bbox_top + bbox_size[1] / 2)
            
            _bbox_size = (max(bbox_size), max(bbox_size))
            _bbox_center = (min(max(_bbox_size[0] / 2, bbox_center[0]), width - _bbox_size[0] /2), 
                            min(max(_bbox_size[1] / 2, bbox_center[1]), height - _bbox_size[1] /2))
            
            _image = image[int(_bbox_center[1]-_bbox_size[1]/2):int(_bbox_center[1] + _bbox_size[1]/2),
                        int(_bbox_center[0]-_bbox_size[0]/2):int(_bbox_center[0] + _bbox_size[0]/2)]
            _mask = mask[int(_bbox_center[1]-_bbox_size[1]/2):int(_bbox_center[1] + _bbox_size[1]/2),
                        int(_bbox_center[0]-_bbox_size[0]/2):int(_bbox_center[0] + _bbox_size[0]/2)]
        else:
            _image = image
            _mask = mask
        
        if fill != "none":
            avgcolor = np.mean((1 - _mask) * _image, (0, 1))
            avgcolor = np.round(avgcolor).astype(np.int32)
            blackcolor = np.array([0, 0, 0])
            fillcolor = blackcolor if fill == "black" else avgcolor
            regionimage = _image * _mask + fillcolor[None, None, :] * (1 - _mask)
        else:
            regionimage = _image
            
        regionimages.append(regionimage)
        regioncaption = regioninfo["caption"]
        regiondescriptions.append(regioncaption)
        
    inputs = processor(text=regiondescriptions, images=regionimages, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    # probs = logits_per_image.softmax(dim=1)
    return logits_per_image, regiondescriptions
    
    
def run2(path="coco2017.4dev.examples.pkl"):
    model, processor = load_clip_model()
    with open(path, "rb") as f:
        loadedexamples = pickle.load(f)
        
    for example in loadedexamples:
        output, _ = do_example(example, model, processor, tightcrop=True, fill="avg")
        print(output.softmax(-1))
        
    print("done")
    
    
def run3(
         path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_bothext2_v4.1_exp_2_forreal/generated_10/",
         tightcrop=True,
         fill="none",
         device=0,
         ):
    device = torch.device("cuda", device)
    batchespath = Path(path) / "outbatches.pkl"
    model, processor = load_clip_model()
    model = model.to(device)
    with open(batchespath, "rb") as f:
        loadedexamples = pickle.load(f)
        
    for batch in loadedexamples:
        for example in batch:
            output, descriptions = do_example(example, model, processor, tightcrop=tightcrop, fill=fill)
            print(descriptions)
            print(output.softmax(-1))
            print("-")
        
    print("done")
    
    
if __name__ == "__main__":
    run3()