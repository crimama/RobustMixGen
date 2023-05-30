import json
import os
import random
import numpy as np 
import cv2     
from torch.utils.data import Dataset
import torch 

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms 
from dataset.utils import pre_caption

class re_train_dataset(Dataset):
    def __init__(self, ann_file,transform,image_root,romixgen,romixgen_true=True,romixgen_ratio=0.1, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
        self.romixgen = romixgen 
        self.romixgen_true = romixgen_true 
        self.romixgen_ratio = romixgen_ratio
        
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        ann = self.ann[index]
        
        if (self.romixgen_true) & (random.random() < self.romixgen_ratio):
            image,caption = self.romixgen(ann)
            
        else:
            image_path = os.path.join(self.image_root,ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            caption = pre_caption(ann['caption'],self.max_words)
        
        return image, caption, self.img_ids[ann['image_id']]

    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
    
from styleformer import Styleformer
class re_eval_perturb_dataset(Dataset):
    def __init__(self, ann_file, img_size, image_root,pertur=None,txt_pertur=None, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.image_root = image_root
        self.max_words = max_words 
        self.img_size = img_size 
        self.pertur = self.pertur_check(pertur)
        self.txt_pertur = self.pertur_check(txt_pertur)
        self.transforms = self.get_transforms()
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(self.txt_pertur(pre_caption(caption,self.max_words)))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def pertur_check(self,pertur):
        def clean(value,**params):
            return value 
        if not pertur:
            return clean 
        else:
            return pertur 
            
    
    
    def get_transforms(self):
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        test_transform      = transforms.Compose([
                                                transforms.Resize((self.img_size,self.img_size),interpolation=Image.BICUBIC),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])
        return test_transform 
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')

        image = self.pertur(np.array(image),severity=2).astype(np.uint8)
        image = self.transforms(Image.fromarray(image).convert('RGB'))
        return image, index
           
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
        
        return image, caption
            

    
