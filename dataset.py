import fiftyone.zoo as foz
from PIL import Image
import json
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset, DataLoader
import tqdm
from transformers import CLIPTokenizer
from torchvision.transforms.functional import to_tensor, to_pil_image
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import itertools
import numpy as np
import colorsys
from einops import rearrange, repeat


class ProcessedCOCOExample(object):
    def __init__(self, img_path, captions, regions, cropsize=None):
        super().__init__()
        self.image_path = img_path
        self.captions = captions
        self.regions = regions
        self.cropsize = cropsize
        
        
def _img_importance_flatten(img: torch.tensor, w: int, h: int) -> torch.tensor:
    return torch.nn.functional.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        # scale_factor=1 / ratio,
        size=(w, h),
        mode="bilinear",
        align_corners=True,
    ).squeeze()
    
    
def colorgen(num_colors=100):
    for i in range(1, num_colors):
        r = (i * 53) % 256  # Adjust the prime number for different color patterns
        g = (i * 97) % 256
        b = (i * 163) % 256
        yield [r/256, g/256, b/256]
        

def colorgen_hsv(numhues=36):
    hue = random.randint(0, 360)
    usehues = set()
    huestep = round(360/numhues)
    retries = 0
    while True:
        sat = random.uniform(0.5, 0.9)
        val = random.uniform(0.3, 0.7)
        yield colorsys.hsv_to_rgb(hue/360, sat, val)
        usehues.add(hue)
        # change hue 
        while hue in usehues:
            hue = (hue + huestep * random.randint(0, int(360/huestep))) % 360
            retries += 1
            if retries > numhues:
                usehues = set()
                retries = 0
                continue
            
            
def randomcolor_hsv():
    hue = random.uniform(0, 360)
    sat = random.uniform(0.5, 0.9)
    val = random.uniform(0.3, 0.7)
    return colorsys.hsv_to_rgb(hue/360, sat, val)
    
    
def materialize_example(example):
    # materialize one example
    # 3. load image
    img = Image.open(example.image_path).convert("RGB")
    imgtensor = to_tensor(img)
    cond_imgtensor = torch.ones_like(imgtensor) * torch.tensor(randomcolor_hsv())[:, None, None]
    
    # 1. pick one caption at random (TODO: or generate one from regions)
    captions = [random.choice(example.captions)[0]]
    # initialize layer ids
    layerids = [torch.zeros_like(captions[0])]
    # 4. load masks
    masks = [torch.ones_like(imgtensor[0], dtype=torch.bool)]
    # 2. get the captions of the regions and build layer ids
    # coloriter = colorgen_hsv()
    for i, region in enumerate(example.regions):
        captions.append(region[1][0])
        layerids.append(torch.ones_like(region[1][0]) * (i + 1))
        masks.append(torch.tensor(region[0]))
        
        randomcolor = torch.tensor(randomcolor_hsv())
        mask = torch.tensor(region[0])
        maskcolor = mask.unsqueeze(0).repeat(3, 1, 1) * randomcolor[:, None, None]
    
        cond_imgtensor = torch.where(mask.unsqueeze(0) > 0.5, maskcolor, cond_imgtensor)
    # finalize captions and layer ids
#         caption, layerids = torch.cat(captions, 0), torch.cat(layerids, 0)

    # random square crop of size divisble by 64 and maximum size 512
    cropsize = min((min(imgtensor[0].shape) // 64) * 64, 512)
    crop = (random.randint(0, imgtensor.shape[1] - cropsize), 
            random.randint(0, imgtensor.shape[2] - cropsize))
    # print(cropsize)
    
    imgtensor = imgtensor[:, crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize]
    cond_imgtensor = cond_imgtensor[:, crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize]
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
    
    # TODO: provide conditioning image based on layers
    
    return {"image": imgtensor, 
            "cond_image": cond_imgtensor,
            "captions": captions,
            "layerids": layerids,
            "regionmasks": downmasktensors
            }
    
    
class COCODatasetSubset(Dataset):
    def __init__(self, examples) -> None:
        super().__init__()
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return materialize_example(self.examples[index])
        

class COCODataset(IterableDataset):
    def __init__(self, split="valid", maxmasks=20, max_samples=100, shuffle=False,
                 captionpath="/USERSPACE/lukovdg1/controlnet11/coco/annotations/", 
                 tokenizer_version="openai/clip-vit-large-patch14"):
        super().__init__()
        self.n = 0
        self.maxmasks = maxmasks
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_version)
        self.shuffle = shuffle
        
        self.captiondb = {}   # from image_id to list of captions
        captionfiles = [captionpath + "captions_val2014.json", captionpath + "captions_train2014.json"]
        for captionfile in captionfiles:
            captions = json.load(open(captionfile))
            for annotation in captions["annotations"]:
                imgid = annotation["image_id"]
                if imgid not in self.captiondb:
                    self.captiondb[imgid] = []
                self.captiondb[imgid].append(annotation["caption"])
                
        if split.startswith("val"):
            segdata = foz.load_zoo_dataset(
                "coco-2017",
                split="validation",
                max_samples=max_samples,
                label_types=["segmentations"],
                shuffle=False,
            )
            segdata.compute_metadata()
        elif split.startswith("tr"):
            segdata = foz.load_zoo_dataset(
                "coco-2017",
                split="train",
                label_types=["segmentations"],
                max_samples=max_samples,
                shuffle=False,
            )
            segdata.compute_metadata()
            
        # self.examples = []
        numtoomanyregions = 0
        
        self.sizestats = {}
        self.examplespersize = {}
        
        for example in tqdm.tqdm(segdata):
            image_path = example.filepath
            image_id = int(Path(example.filepath).stem)

            captions = self.captiondb[image_id]
            captions = [self.tokenize([caption]) for caption in captions]
            frame_size = (example.metadata["width"], example.metadata["height"])
            cropsize = min((min(frame_size) // 64) * 64, 512)
            if cropsize < 350:
                continue
            
            if cropsize not in self.sizestats:
                self.sizestats[cropsize] = 0
            self.sizestats[cropsize] += 1
            
            if example.ground_truth is None:
                continue
                
            regions = []
            # prevent overlapping masks by zeroing out the regions that come later where they overlap with earlier ones
            runningmask = None
            for region in example.ground_truth.detections:
                segmentation = region.to_segmentation(frame_size=frame_size)
                segmask = np.array(segmentation.mask, dtype=bool)
                if runningmask is None:
                    runningmask = np.zeros_like(segmask)
                segmask = segmask & (~runningmask)
                regions.append((segmask, self.tokenize([region.label])))
                runningmask = runningmask | segmask

            if len(regions) > maxmasks:
                numtoomanyregions += 1
                continue
            
            if cropsize not in self.examplespersize:
                self.examplespersize[cropsize] = []
            self.examplespersize[cropsize].append(ProcessedCOCOExample(image_path, captions, regions, cropsize=cropsize))
                
            # self.examples.append(ProcessedCOCOExample(image_path, captions, regions, cropsize=cropsize))     
        
        self.examples = [(k, v) for k, v in self.examplespersize.items()]
        self.examples = sorted(self.examples, key=lambda x: x[0])
        
        self.total_n = sum([len(v) for k, v in self.examples])
            
        # self.examples = sorted(self.examples, key=lambda x: x.cropsize)
            
        # print("Size stats:")
        # print(self.sizestats)
#         print(f"Retained examples: {len(self.examples)}")
#         print(f"Too many regions: {numtoomanyregions}")

    def tokenize(self, x, tokenizer=None):
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        tokenized = tokenizer(x,  # padding="max_length",
                                  max_length=tokenizer.model_max_length,
                                  return_overflowing_tokens=False,
                                  truncation=True,
                                  return_tensors="pt")
        return tokenized["input_ids"]
    
    def untokenize(self, x, tokenizer=None):
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        ret = tokenizer.decode(x)
        return ret
    
    def __iter__(self):
        self.n = 0
        if self.shuffle:
            for k, v in self.examples:
                random.shuffle(v)
        return self
    
    def __next__(self):
        if self.n >= self.total_n:
            raise StopIteration()
        
        prevc, c = 0, 0
        # find bucket
        for k, v in self.examples:
            prevc = c
            c += len(v)
            if prevc <= self.n < c:
                break
        # get example
        example = v[self.n - prevc]
        # increase count
        self.n += 1
        
    # def __getitem__(self, item):
    #     example = self.examples[item]
        return materialize_example(example)
    
    def __len__(self):
        return len(self.examples)
    
    def collate_fn(self, examples):
        # compute size stats
        sizestats = {}
        for example in examples:
            newsize = example["image"].shape[1]
            if newsize not in sizestats:
                sizestats[newsize] = 0
            sizestats[newsize] += 1
        # if sizes are different, throw away those not matching the size of majority
        if len(sizestats) > 1:
            majoritysize, majoritycount = 0, 0
            for s, sc in sizestats.items():
                if sc >= majoritycount:
                    if s > majoritysize:
                        majoritysize, majoritycount = s, sc
                        
            examples = [example for example in examples if example["image"].shape[1] == majoritysize]
        
        # every example is dictionary like specified above
        
        images = []
        cond_images = []
        captions = []
        regionmasks = []
        layerids = []
        regioncounts = []
        
        for example in examples:
            images.append(example["image"])   # concat images
            cond_images.append(example["cond_image"])
            captions.append(torch.cat(example["captions"], 0))   # batchify captions
            regioncounts.append(len(example["captions"]))  # keep track of the number of regions per example
            layerids.append(torch.cat(example["layerids"], 0))   # layer ids
            materialized_masks = {res: masks[layerids[-1]] for res, masks in example["regionmasks"].items()}
            
            regionmasks.append(materialized_masks)
            
        imagebatch = torch.stack(images, dim=0)
        cond_imagebatch = torch.stack(cond_images, dim=0)
        captionbatch = pad_sequence(captions, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        layeridsbatch = pad_sequence(layerids, batch_first=True, padding_value=-1)
        captiontypes = [(layerids_i > 0).long() for layerids_i in layerids]
        captiontypes = pad_sequence(captiontypes, batch_first=True, padding_value=-2)
        captiontypes += 1
        captiontypes[:, 0] = 0
        
        batched_regionmasks = {}
        for regionmask in regionmasks:
            for res, rm in regionmask.items():
                if res not in batched_regionmasks:
                    batched_regionmasks[res] = []
                batched_regionmasks[res].append(rm)
        batched_regionmasks = {res: pad_sequence(v, batch_first=True) for res, v in batched_regionmasks.items()}
        
        # DONE: stack regionmasks to form one tensor (batsize, seqlen, H, W) per mask resolution
        # DONE: passing layer ids: prepare a data structure for converting from current dynamically flat caption format to (batsize, seqlen, hdim)
        # DONE: return (batsize, seqlen) tensor that specifies if the token is part of global description or local description
        # DONE: provide conditioning image for ControlNet
        return {"image": rearrange(imagebatch, 'b c h w -> b h w c'), 
                "cond_image": rearrange(cond_imagebatch, 'b c h w -> b h w c'),
                "caption": captionbatch, 
                "layerids": layeridsbatch, 
                "regionmasks": batched_regionmasks, 
                "captiontypes": captiontypes}
    

class COCODataLoader(object):
    def __init__(self, cocodataset:COCODataset, batch_size=2, shuffle=False, num_workers=0) -> None:
        super().__init__()
        self.ds = cocodataset
        self.dataloaders = []
        for k, cocosubset in self.ds.examples:
            if isinstance(batch_size, dict):
                batsize = batch_size[k]
            else:
                batsize = batch_size
            subdl = DataLoader(COCODatasetSubset(cocosubset), 
                                       batch_size=batsize, 
                                       collate_fn=cocodataset.collate_fn, 
                                       shuffle=shuffle,
                                       num_workers=num_workers)
            self.dataloaders.append(subdl)
        self.subdls_lens = [len(dl) for dl in self.dataloaders]
        print(self.subdls_lens)
        
    def __iter__(self):
        return itertools.chain(*[iter(subdl) for subdl in self.dataloaders])
        
    
def main(x=0):
    cocodataset = COCODataset(max_samples=100)
    print(len(cocodataset))
    
    dl = COCODataLoader(cocodataset, batch_size={384: 5, 448:4, 512: 4}, num_workers=0)
    
    batch = next(iter(dl))
    # print(batch)
    
    for epoch in range(1):
        i = 0
        for batch in dl:
            print(i, batch["image"].shape)
            i += 1
    

if __name__ == "__main__":
    main()