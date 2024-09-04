import copy
import json
from pathlib import Path
import re
from PIL import Image
import requests
import torch

from torchvision.transforms.functional import to_tensor

# from cldm.logger import nested_to
# from dataset import COCOPanopticDataset, COCOPanopticExample, COCODataLoader
import pickle

import numpy as np
import tqdm
import math

import fire



def load_clip_model(modelname="openai/clip-vit-base-patch32"):
    from transformers import CLIPProcessor, CLIPModel
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
    
    
class MetricScore():
    def __init__(self, name, higher_is_better=True):
        self.name = name
        self.higher_is_better = higher_is_better
        
    @classmethod
    def from_string(cls, x):   # must be in format like __str__
        m = re.match(r"Metric\[([^,]+),(.)\]", x)
        if m is not None:
            name, higherbetter = m.groups()
            return cls(name, higherbetter == "+")
        else:
            return None
        
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other):
        return str(self).__eq__(str(other))
    
    def __str__(self):
        return f"Metric[{self.name},{'+' if self.higher_is_better else '-'}]"
    
    def __repr__(self):
        return str(self)
    
    
class DictExampleWrapper:
    def __init__(self, datadict):
        super().__init__()
        self._datadict = datadict
        
    def load_image(self):
        return self._datadict["image"]
    
    def load_seg_image(self):
        return self._datadict["seg_img"]
    
    @property
    def captions(self):
        return [self._datadict["caption"]]
    
    
class LocalCLIPEvaluator():
    LOGITS = MetricScore("localclip_logits")
    COSINE = MetricScore("localclip_cosine")
    PROBS = MetricScore("localclip_probs")
    ACCURACY = MetricScore("localclip_acc")
    
    def __init__(self, clipmodel, clipprocessor, tightcrop=True, fill="none"):
        super().__init__()
        self.clipmodel, self.clipprocessor, self.tightcrop, self.fill = clipmodel, clipprocessor, tightcrop, fill
        
        
    def prepare_example(self, x, tightcrop=None, fill=None):
        if isinstance(x, DictExampleWrapper):
            return self.prepare_example_dictwrapper(x, tightcrop=tightcrop, fill=fill)
        else:
            return self.prepare_example_controlnet(x, tightcrop=tightcrop, fill=fill)
        
    def prepare_example_dictwrapper(self, x, tightcrop=None, fill=None):
        tightcrop = self.tightcrop if tightcrop is None else tightcrop
        fill = self.fill if fill is None else fill
        
        regionimages = []
        regiondescriptions = []
        
        image = np.array(x.load_image())
        
        if "masks" in x._datadict:
            for mask, regioncaption in zip(x._datadict["masks"], x._datadict["prompts"]):
                height, width = mask.shape
            
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
                
                if fill in ("none", "") or fill is not None:
                    regionimage = _image
                else:
                    if fill == "black":
                        fillcolor = np.array([0, 0, 0])
                    elif fill == "white":
                        fillcolor = np.array([1, 1, 1]) * 255
                    else:
                        avgcolor = np.mean((1 - _mask) * _image, (0, 1))
                        avgcolor = np.round(avgcolor).astype(np.int32)
                        fillcolor = avgcolor
                    regionimage = _image * _mask + fillcolor[None, None, :] * (1 - _mask)
            
                regionimages.append(regionimage)
                regiondescriptions.append(regioncaption)            
        
        elif "bboxes" in x._datadict:
            assert fill in ("none", "") or fill is None
            height, width, _ = image.shape
            
            for bbox, regioncaption in zip(x._datadict["bboxes"], x._datadict["bbox_captions"]):
                if tightcrop:
                    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
                    bbox_left, bbox_top, bbox_right, bbox_bottom = \
                        int(bbox_left * width), int(bbox_top * height), int(bbox_right * width), int(bbox_bottom * height)
                
                    bbox_size = ((bbox_right-bbox_left), (bbox_bottom - bbox_top))
                    bbox_center = (bbox_left + bbox_size[0] / 2, bbox_top + bbox_size[1] / 2)
                    
                    _bbox_size = (max(bbox_size), max(bbox_size))
                    _bbox_center = (min(max(_bbox_size[0] / 2, bbox_center[0]), width - _bbox_size[0] /2), 
                                    min(max(_bbox_size[1] / 2, bbox_center[1]), height - _bbox_size[1] /2))
                    
                    _image = image[int(_bbox_center[1]-_bbox_size[1]/2):int(_bbox_center[1] + _bbox_size[1]/2),
                                int(_bbox_center[0]-_bbox_size[0]/2):int(_bbox_center[0] + _bbox_size[0]/2)]
                else:
                    _image = image
                    
                regionimage = _image
                
                regionimages.append(regionimage)
                regiondescriptions.append(regioncaption) 
        
        else:
            pass
            
        return regionimages, regiondescriptions
        
    def prepare_example_controlnet(self, x, tightcrop=None, fill=None):
        tightcrop = self.tightcrop if tightcrop is None else tightcrop
        fill = self.fill if fill is None else fill
        
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
            
            if fill in ("none", "") or fill is not None:
                regionimage = _image
            else:
                if fill == "black":
                    fillcolor = np.array([0, 0, 0])
                elif fill == "white":
                    fillcolor = np.array([1, 1, 1]) * 255
                else:
                    avgcolor = np.mean((1 - _mask) * _image, (0, 1))
                    avgcolor = np.round(avgcolor).astype(np.int32)
                    fillcolor = avgcolor
                regionimage = _image * _mask + fillcolor[None, None, :] * (1 - _mask)
                
            regioncaption = regioninfo["caption"]
            
            # extra region caption from global prompt
            assert len(x.captions) == 1
            caption = x.captions[0]
            matches = re.findall("{([^:]+):" + str(regioncode) + "}", caption)
            assert len(matches) == 1
            regioncaption = matches[0]
            
            regionimages.append(regionimage)
            regiondescriptions.append(regioncaption)
            
        return regionimages, regiondescriptions
    
    def run(self, x):
        regionimages, regiondescriptions = self.prepare_example(x)
            
        inputs = self.clipprocessor(text=regiondescriptions, images=regionimages, return_tensors="pt", padding=True)
        inputs = inputs.to(self.clipmodel.device)
        outputs = self.clipmodel(**inputs)
        logits_per_image = outputs.logits_per_image
        cosine_per_image = (outputs.image_embeds @ outputs.text_embeds.T)
        prob_per_image = logits_per_image.softmax(-1)
        acc_per_image = logits_per_image.softmax(-1).max(-1)[1] == torch.arange(len(logits_per_image), device=logits_per_image.device)
        # probs = logits_per_image.softmax(dim=1)
        return {self.LOGITS: logits_per_image.diag().mean().detach().cpu().item(),
                self.COSINE: cosine_per_image.diag().mean().detach().cpu().item(),
                self.PROBS: prob_per_image.diag().mean().detach().cpu().item(),
                self.ACCURACY: acc_per_image.float().mean().detach().cpu().item()}

LLAVA_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
LLAVA_DEVICE = "cuda:0"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" 
MISTRAL_DEVICE = "cuda:1"

class LLAVACaptioner:
    def __init__(self, llavamodel=LLAVA_MODEL, mistralmodel=MISTRAL_MODEL, llavadevice=LLAVA_DEVICE, mistraldevice=MISTRAL_DEVICE, loadedmodels=None):
        self.mistraldevice = mistraldevice
        self.llavadevice = llavadevice
        if loadedmodels is None:
            loadedmodels = self.loadmodels(llavamodel, mistralmodel, llavadevice, mistraldevice)
        self.llavaprocessor, self.llavamodel, self.mistralprocessor, self.mistralmodel = loadedmodels
        
    def get_loadedmodels(self):
        return self.llavaprocessor, self.llavamodel, self.mistralprocessor, self.mistralmodel
        
    def loadmodels(self, llavamodel, mistralmodel, llavadevice, mistraldevice):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if llavamodel is not None:
            llavaprocessor = LlavaNextProcessor.from_pretrained(llavamodel)
            llavamodel = LlavaNextForConditionalGeneration.from_pretrained(llavamodel, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=llavadevice) 
        else:
            llavaprocessor, llavamodel = None, None
        
        if mistralmodel is not None:
            mistralprocessor = AutoTokenizer.from_pretrained(mistralmodel)
            mistralmodel = AutoModelForCausalLM.from_pretrained(mistralmodel, torch_dtype=torch.float16, device_map=mistraldevice)
        else:
            mistralprocessor, mistralmodel = None, None
        
        return llavaprocessor, llavamodel, mistralprocessor, mistralmodel
    
    def text_to_bits(self, text):
        numvoc = len(self.mistralprocessor.vocab)
        numbitsperword = math.ceil(math.log(numvoc) / math.log(2))
        tokens = self.mistralprocessor(text)["input_ids"][1:]
        bitstr = ""
        for token in tokens:
            tokenstr = f"{token:b}"
            tokenstr = "0"*(numbitsperword - len(tokenstr)) + tokenstr
            bitstr += tokenstr
        return bitstr, tokens
    
    def bits_to_text(self, bits):
        numvoc = len(self.mistralprocessor.vocab)
        numbitsperword = math.ceil(math.log(numvoc) / math.log(2))
        tokens = []
        while len(bits) > 0:
            wordbits = bits[:numbitsperword]
            tokenid = int(wordbits, 2)
            tokens.append(tokenid)
            bits = bits[numbitsperword:]
        text = self.mistralprocessor.decode([1] + tokens, skip_special_tokens=True)
        return text
        
    def describe_image(self, image, short=False, shortlen=16):
        if short:
            prompt = f"[INST] <image> \n Can you please provide a short description of maximum {shortlen} words of the provided image? [/INST]"
        else:
            # prompt = f"[INST] <image> \n Fully describe the object in the image in one phrase. [/INST]"
            prompt = f"[INST] <image> \n What is this object? Fully describe the object in the image in one phrase. [/INST]"
        processed = self.llavaprocessor(prompt, image, return_tensors="pt").to(self.llavadevice)
        # print("input: ", prompt)
        
        out = self.llavamodel.generate(**processed, max_new_tokens=100, pad_token_id=self.llavaprocessor.eos_token_id)

        output = self.llavaprocessor.decode(out[0], skip_special_tokens=True)
        # print("output: ", output)
        
        splits = output.split("[/INST]")
        assert( len(splits) == 2)
        reply = splits[1].strip()
        return reply
    
    def shorten_description(self, text:str, length=14):
        # how = "a comma-separated list of keywords"
        how = "a compact headline"
        # messages = [
        #     {"role": "user", "content": 
        #         f"Can you please rephrase the following image description as {how}, with up to {length} words. Provide just one option. Put most important keywords first. Do not put the result in quotes. The image description is as follows: {description}"}
        # ]
        messages = [
            {"role": "user", "content": 
                f"Gimme a short caption of maximum {length} words from the following image caption. Provide just one option. Do not put the result in quotes. The image caption is as follows: \"{text}\" "}
        ]

        model_inputs = self.mistralprocessor.apply_chat_template(messages, return_tensors="pt").to(self.mistraldevice)

        generated_ids = self.mistralmodel.generate(model_inputs, max_new_tokens=100, do_sample=True)
        output = self.mistralprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        splits = output.split("[/INST]")
        assert( len(splits) == 2)
        reply = splits[1].strip()
        
        # print("Output length: ", len(reply.split()))
        return reply
    
    def reshorten_description(self, text:str, length=14):
        # how = "a comma-separated list of keywords"
        how = "a compact headline"
        # messages = [
        #     {"role": "user", "content": 
        #         f"Can you please rephrase the following image description as {how}, with up to {length} words. Provide just one option. Put most important keywords first. Do not put the result in quotes. The image description is as follows: {description}"}
        # ]
        messages = [
            {"role": "user", "content": 
                f"Can you please make the provided image caption slightly shorter by omitting less important details? Provide just one option. Do not put the result in quotes. The image caption is as follows: \"{description}\" "}
        ]

        model_inputs = self.mistralprocessor.apply_chat_template(messages, return_tensors="pt").to(self.mistraldevice)

        generated_ids = self.mistralmodel.generate(model_inputs, max_new_tokens=100, do_sample=True)
        output = self.mistralprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        splits = output.split("[/INST]")
        assert( len(splits) == 2)
        reply = splits[1].strip()
        
        # print("Output length: ", len(reply.split()))
        return reply
    
    def check_description(self, image, text:str):
            
        # extra = "Remember to look at the different objects and their characteristics (such as colors, shapes). "
        extra = ""
            
        # prompt = f"[INST] <image> \n Does the following description describe the content of the provided image well: \"{text}\" . Answer with \"yes\" or \"no\" only. {extra}[/INST]"
        # prompt = f"[INST] <image> \n Are there any significant differences between the provided image and the following description: \"{text}\"? Answer with \"yes\" or \"no\" only. {extra}[/INST]"
        # prompt = f"[INST] <image> \n Rate the similarity of the provided image with the following description: \"{text}\" . Answer only with a similarity rating between 1 (=lowest similarity) and 5 (=highest similarity). {extra}[/INST]"
        # prompt = f"[INST] <image> \n Does the given image match this description: \"{text}\" ? Answer with \"yes\" or \"no\" only.  [/INST]"
        prompt = f"[INST] <image> \n Is this an image of {text} ? Answer with \"yes\" or \"no\" only.  [/INST]"
        
        processed = self.llavaprocessor(prompt, image, return_tensors="pt").to(self.llavadevice)
        # print("input: ", prompt)
        
        out = self.llavamodel.generate(**processed, max_new_tokens=100, pad_token_id=self.llavaprocessor.tokenizer.eos_token_id)

        output = self.llavaprocessor.decode(out[0], skip_special_tokens=True)
        # print("output: ", output)
        
        splits = output.split("[/INST]")
        assert( len(splits) == 2)
        reply = splits[1].strip()
        return reply
    
    def choose_description(self, image, choices):
        choiceletters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        # choiceletters = [str(i+1) for i in range(10)]
        choiceD = {k: v for k, v in zip(range(len(choiceletters)), choiceletters)}
        revchoiceD = {v: k for k, v in choiceD.items()}
        assert(len(choiceletters) > len(choices))
        
        formattedchoices = zip(choiceletters[:len(choices)], choices)
        formattedchoices = [f"({l}) {c}" for l, c in formattedchoices]
        formattedchoices = ", ".join(formattedchoices)
            
        # prompt = f"[INST] <image> \n Does the following description describe the content of the provided image well: \"{text}\" . Answer with \"yes\" or \"no\" only. {extra}[/INST]"
        # prompt = f"[INST] <image> \n Are there any significant differences between the provided image and the following description: \"{text}\"? Answer with \"yes\" or \"no\" only. {extra}[/INST]"
        # prompt = f"[INST] <image> \n Rate the similarity of the provided image with the following description: \"{text}\" . Answer only with a similarity rating between 1 (=lowest similarity) and 5 (=highest similarity). {extra}[/INST]"
        # prompt = f"[INST] <image> \n Does the given image match this description: \"{text}\" ? Answer with \"yes\" or \"no\" only.  [/INST]"
        prompt = f"[INST] <image> \n Which of the following options describes this image best? {formattedchoices}. Reply with a number only. [/INST]"
        
        processed = self.llavaprocessor(prompt, image, return_tensors="pt").to(self.llavadevice)
        # print("input: ", prompt)
        
        out = self.llavamodel.generate(**processed, max_new_tokens=100, pad_token_id=self.llavaprocessor.tokenizer.eos_token_id)

        output = self.llavaprocessor.decode(out[0], skip_special_tokens=True)
        # print("output: ", output)
        
        splits = output.split("[/INST]")
        assert( len(splits) == 2)
        reply = splits[1].strip()
        
        reply = revchoiceD[reply] if reply in revchoiceD else None
        return reply
    
    def check_description_text(self, image=None, text:str=None, imagetext=None):
        assert text is not None
        textlen = len(text.split())
        if image is None:
            assert imagetext is not None
        else:
            assert imagetext is None
            imagetext = self.describe_image(image, short=True, shortlen=textlen)
        
        rating = self.compare_descriptions(imagetext, text)
        return rating
    
    def compare_descriptions(self, textA, textB):
        messages = [
            {"role": "user", "content": 
                f"Are the following two image captions describing the same image or are there things that are different? First caption is \"{textA}\" and the second caption is \"{textB}\". Answer only with a similarity rating between 1 (=lowest similarity) and 5 (=highest similarity)!"}
        ]

        model_inputs = self.mistralprocessor.apply_chat_template(messages, return_tensors="pt").to(self.mistraldevice)

        generated_ids = self.mistralmodel.generate(model_inputs, max_new_tokens=100, do_sample=True)
        output = self.mistralprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        splits = output.split("[/INST]")
        if len(splits) != 2:
            print(len(splits), output)
        reply = splits[-1].strip()
        return reply
    
    def describe_and_check(self, imagepath, text:str):
        if is_url(imagepath):
            image = Image.open(requests.get(imagepath, stream=True).raw)
        else:
            image = Image.open(imagepath)
            
        # extra = "Remember to look at the different objects and their characteristics (such as colors, shapes). "
        extra = ""
            
        prompt = f"[INST] <image> \n Check and describe the differences between the provided image and the following image description: {text}. Conclude your reply with \"yes\" or \"no\" whether the provided image matches the provided description. [/INST]"
        # prompt = f"[INST] <image> \n Are there any significant differences between the provided image and the following description: \"{text}\"? Answer with \"yes\" or \"no\" only. {extra}[/INST]"
        
        processed = self.llavaprocessor(prompt, image, return_tensors="pt").to(self.llavadevice)
        # print("input: ", prompt)
        
        out = self.llavamodel.generate(**processed, max_new_tokens=100)

        output = self.llavaprocessor.decode(out[0], skip_special_tokens=True)
        # print("output: ", output)
        
        splits = output.split("[/INST]")
        assert( len(splits) == 2)
        reply = splits[1].strip()
        return reply
    
    def explain_difference(self, imagepath, text:str):
        if is_url(imagepath):
            image = Image.open(requests.get(imagepath, stream=True).raw)
        else:
            image = Image.open(imagepath)

        # extra = "Remember to look at the different objects and their characteristics (such as colors, shapes). "
        extra = ""            
        
        prompt = f"[INST] <image> \n Does the following description describe the provided image well: \"{text}\" . Make a list of things you checked and elaborate. Conclude your reply with a clear \"yes\" or \"no\". {extra} [/INST]"
        # prompt = f"[INST] <image> \n Are there any significant differences between the provided image and the following description: \"{text}\"? Answer with \"yes\" or \"no\" and make a list of things you checked and elaborate. {extra}[/INST]"
        
        processed = self.llavaprocessor(prompt, image, return_tensors="pt").to(self.llavadevice)
        # print("input: ", prompt)
        
        out = self.llavamodel.generate(**processed, max_new_tokens=200)

        output = self.llavaprocessor.decode(out[0], skip_special_tokens=True)
        # print("output: ", output)
        
        splits = output.split("[/INST]")
        assert( len(splits) == 2)
        reply = splits[1].strip()
        return reply
    
    def check_description_explain(self, imagepath, text:str):
        diff = self.explain_difference(imagepath, text)
        
        messages = [
            {"role": "user", "content": 
                f"Please answer with \"yes\" or \"no\" whether the following paragraph says the image matches the description or not: \"{diff}\". Limit your reply to one word only: either \"Yes\" or \"No\"!"}
        ]

        model_inputs = self.mistralprocessor.apply_chat_template(messages, return_tensors="pt").to(self.mistraldevice)

        generated_ids = self.mistralmodel.generate(model_inputs, max_new_tokens=100, do_sample=True)
        output = self.mistralprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        splits = output.split("[/INST]")
        if len(splits) != 2:
            print(len(splits), output)
        reply = splits[-1].strip()
        return reply, diff
        
        

class LLAVAEvaluator():
    LLAVASCORE = MetricScore("llava_score")
    
    def __init__(self, llavadevice=LLAVA_DEVICE, mistraldevice=MISTRAL_DEVICE, tightcrop=True, fill="none"):
        super().__init__()
        self.captioner = LLAVACaptioner(llavadevice=llavadevice, mistraldevice=mistraldevice, mistralmodel=None)
        self.tightcrop, self.fill = tightcrop, fill
        
    def prepare_example(self, x, tightcrop=None, fill=None):
        if isinstance(x, DictExampleWrapper):
            return self.prepare_example_dictwrapper(x, tightcrop=tightcrop, fill=fill)
        else:
            return self.prepare_example_controlnet(x, tightcrop=tightcrop, fill=fill)
        
    def seg_image_to_mask_and_bboxes(self, pilimg):
        img = (to_tensor(pilimg) * 256).long()
        cimg = img[0] + img[1] * 256 + img[2] * 256 *256
        masks = []
        bboxes = []
        for code in cimg.unique():
            mask = cimg == code
            masks.append(mask)
            masknz = mask.nonzero()
            bbox = (masknz.min(0)[0], masknz.max(0)[0])
            bbox = torch.cat(bbox).float()
            bbox[0], bbox[2] = bbox[0] / img.shape[1], bbox[2] / img.shape[1]
            bbox[1], bbox[3] = bbox[1] / img.shape[2], bbox[3] / img.shape[2]
            bbox = torch.stack([bbox[1], bbox[0], bbox[3], bbox[2]])
            bboxes.append(bbox.numpy())
        return masks, bboxes
    
    def find_closest_mask(self, keybbox, bboxes, masks):
        closest = None
        dist = 1e6
        for bbox, mask in zip(bboxes, masks):
            d = np.linalg.norm(keybbox - bbox)
            if d < dist:
                dist = d
                closest = (bbox, mask)
        return closest
        
    def prepare_example_dictwrapper(self, x, tightcrop=None, fill=None):
        tightcrop = self.tightcrop if tightcrop is None else tightcrop
        fill = self.fill if fill is None else fill
        
        regionimages = []
        regiondescriptions = []
        
        image = np.array(x.load_image())
        
        if "masks" in x._datadict:
            for mask, regioncaption in zip(x._datadict["masks"], x._datadict["prompts"]):
                height, width = mask.shape
            
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
                
                if fill in ("none", "") or fill is None:
                    regionimage = _image
                else:
                    if fill == "black":
                        fillcolor = np.array([0, 0, 0]).astype(np.uint8)
                    elif fill == "white":
                        fillcolor = (np.array([1, 1, 1]) * 255).astype(np.uint8)
                    else:
                        avgcolor = np.mean((1 - _mask) * _image, (0, 1))
                        avgcolor = np.round(avgcolor).astype(np.int32)
                        fillcolor = avgcolor
                    regionimage = np.where(_mask[:, :, None] == 1, _image, fillcolor[None, None, :]).astype(np.uint8)
            
                regionimages.append(regionimage)
                regiondescriptions.append(regioncaption)            
        
        elif "bboxes" in x._datadict:
            # assert fill in ("none", "") or fill is None
            height, width, _ = image.shape
            seg_masks, seg_bboxes = self.seg_image_to_mask_and_bboxes(x._datadict["seg_img"])
            
            for bbox, regioncaption in zip(x._datadict["bboxes"], x._datadict["bbox_captions"]):
                _, mask = self.find_closest_mask(np.array(bbox), seg_bboxes, seg_masks)
                if tightcrop:
                    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
                    bbox_left, bbox_top, bbox_right, bbox_bottom = \
                        int(bbox_left * width), int(bbox_top * height), int(bbox_right * width), int(bbox_bottom * height)
                
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
                      
                if fill in ("none", "") or fill is None:
                    regionimage = _image
                else:
                    if fill == "black":
                        fillcolor = np.array([0, 0, 0]).astype(np.uint8)
                    elif fill == "white":
                        fillcolor = (np.array([1, 1, 1]) * 255).astype(np.uint8)
                    else:
                        avgcolor = np.mean((1 - _mask) * _image, (0, 1))
                        avgcolor = np.round(avgcolor).astype(np.int32)
                        fillcolor = avgcolor
                    regionimage = np.where(_mask[:, :, None] == 1, _image, fillcolor[None, None, :]).astype(np.uint8)
            
                
                regionimages.append(regionimage)
                regiondescriptions.append(regioncaption) 
        
        else:
            pass
            
        return regionimages, regiondescriptions
        
    def prepare_example_controlnet(self, x, tightcrop=None, fill=None):
        tightcrop = self.tightcrop if tightcrop is None else tightcrop
        fill = self.fill if fill is None else fill
        
        regionimages = []
        regiondescriptions = []
        
        image = np.array(x.load_image())
        seg_image = np.array(x.load_seg_image())
        seg_image_codes = rgb_to_regioncode(*np.split(seg_image, 3, -1))
        for regioncode, regioninfo in x.seg_info.items():
            
            # extra region caption from global prompt
            regioncaption = regioninfo["caption"]
            assert len(x.captions) == 1
            caption = x.captions[0]
            matches = re.findall("{([^:]+):" + str(regioncode) + "}", caption)
            if len(matches) == 0:
                continue
            regioncaption = matches[0]
            
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
            
            if fill in ("none", "") or fill is None:
                regionimage = _image
            else:
                if fill == "black":
                    fillcolor = np.array([0, 0, 0]).astype(np.uint8)
                elif fill == "white":
                    fillcolor = (np.array([1, 1, 1]) * 255).astype(np.uint8)
                else:
                    avgcolor = np.mean((1 - _mask) * _image, (0, 1))
                    avgcolor = np.round(avgcolor).astype(np.int32)
                    fillcolor = avgcolor
                regionimage = np.where(_mask == 1, _image, fillcolor[None, None, :]).astype(np.uint8)
                  
            
            regionimages.append(regionimage)
            regiondescriptions.append(regioncaption)
            
        return regionimages, regiondescriptions
    
    def run(self, x):
        regionimages, regiondescriptions = self.prepare_example(x)
        metrics = {}
        
        llavaout = []
        for regionimage in regionimages:
            llavaout.append(self.captioner.choose_description(regionimage, regiondescriptions))
            
        multiple_choice_acc = sum([pred == truth for pred, truth in zip(llavaout, range(len(llavaout)))]) / len(llavaout)
        metrics.update({MetricScore("LLAVA_MULTI_ACC".lower()): multiple_choice_acc})
        
        matrix = np.zeros((len(regionimages), len(regiondescriptions)))
        
        llavadescriptions = []
        for i, regionimage in enumerate(regionimages):
            llavadescriptions.append({})
            for j, regiondescription in enumerate(regiondescriptions):
                reply = self.captioner.check_description(regionimage, regiondescription)
                llavadescriptions[-1][regiondescription] = reply
                matrix[i, j] = reply_to_bool(reply)
                
        metrics.update(compute_llava_scores_from_matrix(matrix))
        
            
        return metrics
    
    
def reply_to_bool(reply):
    if len(reply.strip()) < 2:
        return None
    if reply.strip()[:2].lower().strip() == "no":
        return False
    elif reply.strip()[:3].lower().strip() == "yes":
        return True
    else:
        return None
    
    
def compute_llava_scores_from_matrix(m):
    # accuracy
    ref = np.eye(m.shape[0], m.shape[1]).astype(m.dtype)
    acc = (ref == m).sum() / np.prod(m.shape)
    
    tp = np.diag(m).sum()
    fp = m.sum() - tp
    tn = ((m == ref) & (ref == 0)).sum()
    fn = (m == 0).sum() - tn
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return {MetricScore("LLAVA_BIN_ACC".lower()): acc, MetricScore("LLAVA_PRECISION".lower()): precision, MetricScore("LLAVA_RECALL".lower()): recall}


class AestheticsPredictor():
    SCORE = MetricScore("laion_aest_score")
    
    weight_url_dict = {
        "openai/clip-vit-base-patch32": "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth",
        "openai/clip-vit-large-patch14": "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth",
    }
    hf_clipname_to_openclip = {
        "openai/clip-vit-base-patch32": ("openai", "ViT-B-32"),
        "openai/clip-vit-large-patch14": ("openai", "ViT-L-14"),
    }
    
    def __init__(self, clipname="openai/clip-vit-large-patch14", weightdir="extramodels/aesthetics_predictor", device=torch.device("cuda")):
    # def __init__(self, clipmodel, clipprocessor, weightdir="extramodels/aesthetics_predictor"):
        super().__init__()
        import open_clip
        cliptrainer, clipmodelname = self.hf_clipname_to_openclip[clipname]
        self.clipmodel, _, self.clipprocess = open_clip.create_model_and_transforms(clipmodelname, pretrained=cliptrainer)
        self.clipmodel.to(device)
        weighturl = self.weight_url_dict[clipname]
        self.device = device
        # self.clipmodel, self.clipprocessor = clipmodel, clipprocessor
        # weighturl = self.weight_url_dict[self.clipmodel.name_or_path]
        weightpath = Path(weightdir) / Path(weighturl).name
        if not weightpath.exists():
            # download weights
            weightpath.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(weighturl)
            with open(weightpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        
        assert weightpath.exists()
        if clipname == "openai/clip-vit-large-patch14":
        # if self.clipmodel.name_or_path == "openai/clip-vit-large-patch14":
            self.m = torch.nn.Linear(768, 1)
        else:
            self.m = torch.nn.Linear(512, 1)
        self.m.load_state_dict(torch.load(weightpath))
        self.m.eval()
        self.m.to(self.device)
        
    def run_image(self, x:Image.Image):
        image_input = self.clipprocess(x).unsqueeze(0).to(self.device)
        image_features = self.clipmodel.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = self.m(image_features)
        # image = np.array(x)
        # inputs = self.clipprocessor(text=[""], images=[image], return_tensors="pt", padding=True)
        # inputs = inputs.to(self.clipmodel.device)
        # outputs = self.clipmodel(**inputs)
        # image_embed = outputs.image_embeds
        # image_embed /= image_embed.norm(dim=-1, keepdim=True)
        # prediction = self.m(image_embed)
        assert prediction.shape == (1, 1)
        return prediction[0, 0].cpu().item()
        
    def run(self, x):
        image = x.load_image()
        return {self.SCORE: self.run_image(image), }
    

class MANIQAEvaluator():
    SCORE = MetricScore("maniqa")
    def __init__(self, device=torch.device("cuda"), **kw):
        import pyiqa
        super().__init__(**kw)
        self.metric = pyiqa.create_metric("maniqa", device=device, as_loss=False)
        
    def run(self, x):
        image = x.load_image()
        score = self.metric(image)
        assert score.shape == (1, 1)
        return {self.SCORE: score[0, 0].detach().cpu().item()}
    

class BRISQUEEvaluator():
    SCORE = MetricScore("brisque", higher_is_better=False)
    def __init__(self, device=torch.device("cuda"), **kw):
        import pyiqa
        super().__init__(**kw)
        self.metric = pyiqa.create_metric("brisque", device=device, as_loss=False)
        
    def run(self, x):
        image = x.load_image()
        score = self.metric(image)
        assert score.shape == (1,)
        return {self.SCORE: score[0].detach().cpu().item()}
        
    
def do_example(x, evaluators):
    if isinstance(x, dict):
        x = DictExampleWrapper(x)
    ret = {}
    for evaluator in evaluators:
        if evaluator is not None:
            resultdic = evaluator.run(x)
            for k, v in resultdic.items():
                ret[k] = v
    return ret

    
def run2(path="coco2017.4dev.examples.pkl"):
    model, processor = load_clip_model()
    with open(path, "rb") as f:
        loadedexamples = pickle.load(f)
        
    all_logits, all_cosines, all_probs, all_accs = [], [], [], []
        
    for example in loadedexamples:
        logits, cosines, probs, accs, descriptions = do_example(example, model, processor, tightcrop=True, fill="avg")
        all_logits.append(logits.mean().cpu().item())
        all_cosines.append(cosines.mean().cpu().item())
        all_probs.append(probs.mean().cpu().item())
        all_accs.append(accs.mean().cpu().item())
        
    print(all_accs)
    print("done")
    
    
def load_everything(clip_version="openai/clip-vit-large-patch14", device=0, tightcrop=True, fill="none"):
    # print("loading clip model")
    # model, processor = load_clip_model(modelname=clip_version)
    # model = model.to(device)
        
    print("loading models for evaluation")
    ret = []
    
    # ret.append(LocalCLIPEvaluator(model, processor, tightcrop=tightcrop, fill=fill))
    ret.append(LLAVAEvaluator(tightcrop=tightcrop, fill=fill))
    # ret.append(AestheticsPredictor(clip_version, device=device))
    # ret.append(maniqaevaluator = MANIQAEvaluator(device=device))
    # ret.append(BRISQUEEvaluator(device=device))
    
    print("models loaded")
    return ret
    
    
def run3(
         path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_global_v5_exp_1/generated_extradev.pkl_1",
         tightcrop=True,
         fill="none",
         device=0,
        #  clip_version="openai/clip-vit-base-patch32",
         clip_version="openai/clip-vit-large-patch14",
         evalmodels=None,
         save_suffix="llava",
         ):
    device = torch.device("cuda", device)
    batchespath = Path(path) / "outbatches.pkl"
    
    with open(batchespath, "rb") as f:
        loadedexamples = pickle.load(f)
        
    if evalmodels is None:
        evalmodels = load_everything(clip_version=clip_version, device=device, tightcrop=tightcrop, fill=fill)
        
    # clipevaluator, aestheticspredictor, maniqaevaluator, brisque = evalmodels
    
    colnames = None
    higherbetter = []
    allmetrics = []
        
    with torch.no_grad():
        print("iterating over batches")
        for batch in tqdm.tqdm(loadedexamples):
            batch_metrics = []
            for example in batch:       # all examples in one batch have been generated as different seeds of the same starting point
                outputs = do_example(example, evalmodels)
                outputcolnames, outputdata = zip(*sorted(outputs.items(), key=lambda x: x[0].name))
                if colnames is None:
                    colnames = copy.deepcopy(outputcolnames)
                assert colnames == outputcolnames
                batch_metrics.append(tuple(outputdata))
                # local_clip_metrics = outputs["clipevaluator"]      # metrics here are over regions in this one example
                # example_metrics = [metric.mean().cpu().item() for metric in local_clip_metrics]
                # example_metrics.append(outputs["aestheticspredictor"])
                # example_metrics.append(outputs["maniqa"])
                # example_metrics.append(outputs["brisque"])
                # batch_metrics.append(tuple(example_metrics))
            allmetrics.append(batch_metrics)        # aggregate over all batches --> (numbats, numseeds, nummetrics)
    
    tosave = {"colnames": [str(colname) for colname in colnames],
              "data": allmetrics}
    with open(Path(path) / f"evaluation_results_raw_{save_suffix}.json", "w") as f:
        json.dump(tosave, 
                  f, 
                  indent=4)
    print(f"saved raw results in {Path(path)}")
            
    allmetrics = np.array(allmetrics)
        
    # compute averages per seed --> (nummetrics, numseeds,)
    means_per_seed = allmetrics.mean(0).T
    
    means_over_seeds = means_per_seed.mean(1)
    stds_over_seeds = means_per_seed.std(1)
    
    higher_is_better = np.array([True if colname.higher_is_better else False for colname in colnames])
    max_over_seeds = allmetrics.max(1).mean(0)
    min_over_seeds = allmetrics.min(1).mean(0)
    best_over_seeds = np.where(higher_is_better, max_over_seeds, min_over_seeds)
    
    means_over_seeds_dict = dict(zip(colnames, list(means_over_seeds)))
    stds_over_seeds_dict = dict(zip(colnames, list(stds_over_seeds)))
    best_over_seeds_dict = dict(zip(colnames, list(best_over_seeds)))
        
    print(means_over_seeds_dict)
    print(stds_over_seeds_dict)
    print(best_over_seeds_dict)
    
    means_over_seeds_dict = {str(k): v for k, v in means_over_seeds_dict.items()}
    stds_over_seeds_dict = {str(k): v for k, v in stds_over_seeds_dict.items()}
    best_over_seeds_dict = {str(k): v for k, v in best_over_seeds_dict.items()}
    
    tosave = {     "means": means_over_seeds_dict, 
                   "stds": stds_over_seeds_dict, 
                   "best": best_over_seeds_dict, 
                #    "alldata": allmetrics, 
                #    "colnames": colnames
             } 
    
    with open(Path(path) / f"evaluation_results_summary_{save_suffix}.json", "w") as f:
        json.dump(tosave, 
                  f, 
                  indent=4)
    print(json.dumps(tosave, indent=4))
    print(f"saved in {Path(path)}")
    
    
def tst_aesthetics():
    image = Image.open("lovely-cat-as-domestic-animal-view-pictures-182393057.jpg")
    
    aestheticspredictor = AestheticsPredictor()
    score = aestheticspredictor.run_image(image)
    
    print("aesthetic score:", score)
    
    
def run4(
        paths=[
            "/USERSPACE/lukovdg1/DenseDiffusion/gligen_outputs/with_bgr/tau=1.0/*",
            # "/USERSPACE/lukovdg1/DenseDiffusion/dd_outputs/*",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_global_v5_exp_1/*",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_cac_v5_exp_1/*",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_dd_v5_exp_1/*",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_2/*",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_2/*",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_4/*",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_1/*",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_1/generated_threeorange1.pkl_2",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_1/generated_extradev.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_1/generated_catdog.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_1/generated_zebra.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_1/generated_threeorange1.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_2/generated_threeorange1.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_3/generated_threeorange1.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_4/generated_threeorange1.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_5/generated_threeorange1.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_6/generated_threeorange1.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_1/generated_extradev.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_2/generated_extradev.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_3/generated_extradev.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_4/generated_extradev.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_5/generated_extradev.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_6/generated_extradev.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_dd_v5_exp_1/generated_threeorange1.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_dd_v5_exp_2/generated_threeorange1.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_dd_v5_exp_3/generated_threeorange1.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_dd_v5_exp_1/generated_extradev.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_dd_v5_exp_2/generated_extradev.pkl_1",
            # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_dd_v5_exp_3/generated_extradev.pkl_1",
        ],
        tightcrop=True,
        fill="black",
        device=0,
        #  clip_version="openai/clip-vit-base-patch32",
        clip_version="openai/clip-vit-large-patch14",
        sim=False,
    ):
    print("loading everything")
    print(paths)
    if not sim:
        evalmodels = load_everything(clip_version=clip_version, device=device, tightcrop=tightcrop, fill=fill)
    
    totalcount = 0
    for i, path in enumerate(paths):
        print(path)
        assert path.startswith("/")
        subpaths = list(Path("/").glob(path[1:]))
        for j, subpath in enumerate(subpaths):
            if subpath.is_dir() and (subpath / "outbatches.pkl").exists():
                totalcount += 1
                print(f"Doing {subpath} ({i+1}/{len(paths)} path, {j+1}/{len(subpaths)} subpath) (total: {totalcount})")
                if not sim:
                    # try:
                    run3(path=subpath, tightcrop=tightcrop, fill=fill, device=device, clip_version=clip_version, evalmodels=evalmodels)
                    # except Exception as e:
                    #     print("Exception occurred")
                    #     pass
                
    
if __name__ == "__main__":
    # tst_aesthetics()
    # run3()
    fire.Fire(run4)