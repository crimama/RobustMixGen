import json
import os
import random 
import numpy as np 
from PIL import Image
from dataset.utils import pre_caption
from torch.utils.data import Dataset
import torchvision.transforms as transforms 

class ve_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.labels = {'entailment':2,'neutral':1,'contradiction':0}
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,'%s.jpg'%ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(ann['sentence'], self.max_words)

        return image, sentence, self.labels[ann['label']]
    
class ve_pertur_dataset(Dataset):
    def __init__(self, ann_file, img_size, image_root, img_pertur=None, txt_pertur=None, max_words=30):
        self.ann = json.load(open(ann_file,'r'))
        self.img_size   = img_size
        self.transform  = self.get_transform()
        self.image_root = image_root
        self.img_pertur = self.pertur_check(img_pertur)
        self.txt_pertur = self.pertur_check(txt_pertur)
        self.max_words = max_words
        self.labels = {'entailment':2,'neutral':1,'contradiction':0}

        
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
        
        image_path = os.path.join(self.image_root,'%s.jpg'%ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.img_pertur(np.array(image),severity=2).astype(np.uint8)
        image = self.transform(Image.fromarray(image).convert('RGB'))  

        sentence = pre_caption(ann['sentence'], self.max_words)
        sentence = self.txt_pertur(sentence)

        return image, sentence, self.labels[ann['label']]
    