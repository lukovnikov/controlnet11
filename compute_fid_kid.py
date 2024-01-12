from typing import List
import torch_fidelity as tfid
from cleanfid import fid
import fire
from pathlib import Path
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import CenterCrop, Resize, ToTensor
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch


class MaximumCenterCrop(torch.nn.Module):
    def forward(self, img):
        size = min(img.size)
        return center_crop(img, size)
    
    
class ImageFolderDataset(Dataset):
    def __init__(self, path, transform:List=None) -> None:
        super().__init__()
        self.path = path
        self.imgpaths = []
        self.transform = transform
        self.totensor = ToTensor()
        
        if not isinstance(self.transform, list):
            self.transform = [self.transform]
            
        for imgpath in Path(self.path).glob("*"):
            self.imgpaths.append(imgpath)
        
    def __len__(self):
        return len(self.imgpaths)
    
    def __getitem__(self, index) -> Image:
        img = Image.open(self.imgpaths[index])
        for transform_op in self.transform:
            img = transform_op(img)
        img = self.totensor(img)
        img = (img * 255).to(torch.uint8)
        return img


def main(reference_path="/USERSPACE/lukovdg1/coco2017/val2017",
         jpegq=96, resize_size=224,
        #  generated_path="/USERSPACE/lukovdg1/DenseDiffusion/gligen_outputs/with_bgr/tau=1.0/coco2017val_1_out",
        #  generated_path="/USERSPACE/lukovdg1/DenseDiffusion/dd_outputs/coco2017val_4_out",
        #  generated_path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_5/generated_coco2017val_1",
        #  generated_path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_global_v5_exp_1/generated_coco2017val_1",
        # generated_path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_cac_v5_exp_1/generated_coco2017val_1",
        # generated_path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_dd_v5_exp_1/generated_coco2017val_1",
        # generated_path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_2/generated_coco2017val_1",
        # generated_path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_4/generated_coco2017val_1",
        generated_path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_2/generated_coco2017val_1",
         ):
    # SWAP for test
    a = reference_path
    reference_path = generated_path
    generated_path = a
    
    # if not exists, center-crop dataset
    refpath = Path(reference_path)
    croppedpath = Path(reference_path + f"_cropped_{resize_size}")
    if not croppedpath.exists():
        print("Center-cropping and resizing.")
        # do crop
        for imgpath in tqdm(refpath.glob("*")):
            img = Image.open(imgpath)
            cropped_image = center_crop(img, min(img.size))
            cropped_image = resize(cropped_image, resize_size)
            if not croppedpath.exists():
                croppedpath.mkdir(parents=True, exist_ok=True)
            cropped_image.convert("RGB").save(croppedpath / (str(imgpath.stem) + ".jpg"), quality=jpegq)
    else:
        print("Found center-cropped reference images.")
        
    genpath = Path(generated_path)
    croppedpath2 = Path(generated_path + f"_cropped_{resize_size}")
    if not croppedpath2.exists():
        print("Center-cropping")
        # do crop
        for imgpath in tqdm(genpath.glob("*")):
            if imgpath.is_dir() or imgpath.suffix == ".json":
                continue
            img = Image.open(imgpath)
            cropped_image = center_crop(img, min(img.size))
            cropped_image = resize(cropped_image, resize_size)
            if not croppedpath2.exists():
                croppedpath2.mkdir(parents=True, exist_ok=True)
            cropped_image.convert("RGB").save(croppedpath2 / (str(imgpath.stem) + ".jpg"), quality=jpegq)
    else:
        print("Found center-cropped reference images.")
    
    # # check and compute dataset stats:
    # if not fid.test_stats_exists(str(croppedpath), mode="clean"):
    #     print("Computing cropped reference stats")
    #     fid.make_custom_stats(str(croppedpath), str(croppedpath), mode="clean")
    #     print("done.")
    # else:
    #     print(f"Custom stats found for {str(croppedpath)}.")
    
    clean_fid = fid.compute_fid(str(croppedpath), str(croppedpath2))
    clean_kid = fid.compute_kid(str(croppedpath), str(croppedpath2))
    clean_clipfid = fid.compute_fid(str(croppedpath), str(croppedpath2), mode="clean", model_name="clip_vit_b_32")
    
    print(f"Clean FID: {clean_fid}")
    print(f"Clean KID: {clean_kid}")
    print(f"Clean CLIP-FID: {clean_clipfid}")
    
    metrics = tfid.calculate_metrics(
        input1=str(croppedpath),
        input2=str(croppedpath2),
        cuda=True,
        fid=True,
        kid=True,
    )
    
    print(metrics)
    
    
if __name__ == "__main__":
    fire.Fire(main)