import random 
import json
import os
import numpy as np 
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption
from torchvision import transforms 

class nlvr_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root,
                 romixgen: object = None, romixgen_true: bool = False, romixgen_prob: float = 0.0,
                 img_pertur=None, txt_pertur=None):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = 30
        
        self.romixgen = romixgen 
        self.romixgen_true = romixgen_true 
        self.romixgen_prob = romixgen_prob 
        
        self.img_pertur = img_pertur 
        self.txt_pertur = txt_pertur 
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        ann = self.ann[index]
        
        if (self.romixgen_true) & (random.random() < self.romixgen_prob):
            image0, image1, sentence, label = self.romixgen(ann)
            
        else:
            image0_path = os.path.join(self.image_root,ann['images'][0])        
            image1_path = os.path.join(self.image_root,ann['images'][1])              
            
            image0 = Image.open(image0_path).convert('RGB')   
            image1 = Image.open(image1_path).convert('RGB')     
            
            if self.img_pertur:
                image0 = self.img_pertur(np.array(image0),severity=2).astype(np.uint8)
                image1 = self.img_pertur(np.array(image1),severity=2).astype(np.uint8)
                
            image0 = self.transform(image0)   
            image1 = self.transform(image1)          

            sentence = pre_caption(ann['sentence'], self.max_words)
            if self.txt_pertur:
                sentence = self.txt_pertur(sentence)
            
        if ann['label']=='True':
            label = 1
        else:
            label = 0
    
        return image0, image1, sentence, label
    
class nlvr_pertur_dataset(Dataset):
    def __init__(self, ann_file, img_size, image_root, img_pertur=None, txt_pertur=None, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.img_size   = img_size
        self.transform  = self.get_transform()
        self.image_root = image_root
        self.img_pertur = self.pertur_check(img_pertur)
        self.txt_pertur = self.pertur_check(txt_pertur)
        self.max_words = max_words
        

        
    def pertur_check(self, pertur):
        def clean(value, **params):
            return value 
        if not pertur:
            return clean 
        else:
            return pertur 
        
    def __len__(self):
        return len(self.ann)
            
    def get_transform(self):
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        test_transform      = transforms.Compose([
                                                transforms.Resize((self.img_size,self.img_size),interpolation=Image.BICUBIC),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])
        return test_transform
        
            
    def __getitem__(self, index):    
        ann = self.ann[index]
                
        image0_path = os.path.join(self.image_root,ann['images'][0])        
        image1_path = os.path.join(self.image_root,ann['images'][1])              
        
        image0 = Image.open(image0_path).convert('RGB')   
        image1 = Image.open(image1_path).convert('RGB')         

        sentence = pre_caption(ann['sentence'], self.max_words)
            
        if ann['label']=='True':
            label = 1
        else:
            label = 0
            
        # perturbation 
        image0 = self.img_pertur(np.array(image0),severity=2).astype(np.uint8)
        image1 = self.img_pertur(np.array(image1),severity=2).astype(np.uint8)
        
        image0 = self.transform(Image.fromarray(image0).convert('RGB'))  
        image1 = self.transform(Image.fromarray(image1).convert('RGB'))  
        
        sentence = self.txt_pertur(sentence)
    
        return image0, image1, sentence, label