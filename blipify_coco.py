from pathlib import Path
import pickle
import requests
from PIL import Image
import fire
import numpy as np
import torch

from evaluation import prepare_example, rgb_to_regioncode


def load_blip_model(modelname="Salesforce/blip-image-captioning-base"):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained(modelname)
    model = BlipForConditionalGeneration.from_pretrained(modelname).to("cuda")
    return model, processor


def load_blip2_model(modelname="Salesforce/blip2-opt-2.7b"):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained(modelname)
    # model = Blip2ForConditionalGeneration.from_pretrained(modelname, torch_dtype=torch.float16)
    model = Blip2ForConditionalGeneration.from_pretrained(modelname, torch_dtype=torch.float16)
    return model, processor


def do_example2(x, model, processor, tightcrop=True, fill="none"):
    regionimages, regiondescriptions = prepare_example(x, tightcrop=tightcrop, fill=fill)
    blipcaptions = []
    for regionimage, regiondescription in zip(regionimages, regiondescriptions):
        regionimg = Image.fromarray(regionimage.astype('uint8'), 'RGB')
        # prompt = f"Question: Describe in detail with colors the {regiondescription} in this image. Answer:"
        prompt = f"Question: What color does the {regiondescription} have in this image. Answer:"
        # print(prompt)
        inputs = processor(images=regionimg, text=prompt, return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # print(generated_text)
        blipcaptions.append(generated_text)
    return regionimages, blipcaptions


def do_example(x, model, processor, tightcrop=True, fill="none"):
    regionimages, regiondescriptions = prepare_example(x, tightcrop=tightcrop, fill=fill)
    blipcaptions = []
    for regionimage in regionimages:
        regionimg = Image.fromarray(regionimage.astype('uint8'), 'RGB')
        inputs = processor(regionimg, return_tensors="pt").to(model.device)
        out = model.generate(**inputs)
        outstr = processor.decode(out[0], skip_special_tokens=True)
        blipcaptions.append(outstr)
    return regionimages, blipcaptions


def run3(
         path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v4.1/checkpoints_coco_bothext2_v4.1_exp_2_forreal/generated_10/",
         tightcrop=True,
         fill="white",
         device=0,
         use_blip2=True,
         ):
    device = torch.device("cuda", device)
    batchespath = Path(path) / "outbatches.pkl"
    
    model, processor = load_blip_model()
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


# def main():
#     print("loading model")
#     model, processor = load_blip_model()
#     print("loaded model")
    
#     img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
#     img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     img_url = "https://i.imgur.com/dyXPoX1.jpeg"
#     raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

#     # # conditional image captioning
#     # text = "a photo of"
#     # inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

#     # out = model.generate(**inputs)
#     # print(processor.decode(out[0], skip_special_tokens=True))
#     # # >>> a photography of a woman and her dog

#     # unconditional image captioning
#     inputs = processor(raw_image, return_tensors="pt").to("cuda")

#     out = model.generate(**inputs)
#     print(processor.decode(out[0], skip_special_tokens=True))
#     # >>> a woman sitting on the beach with her dog
    
# if __name__ == "__main__":
#     fire.Fire(main)
