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
    def __init__(self, ann_file: str, transform: list, image_root: str, romixgen: object, 
                romixgen_true: bool = True ,romixgen_prob: float = 0.1, max_words: int = 30, dataset: str = 'coco'):        
        '''
        ann_file : annotation file : [{'caption': 'A woman wearing a net on her head cutting a cake. ',
                                        'image ': 'COCO_val2014_000000522418.jpg',
                                        'image_id': 'coco_522418'},...
        
        '''
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
            
        self.transform = transform # transform for nomral image 
        self.image_root = image_root 
        self.max_words = max_words
        
        self.romixgen = romixgen 
        self.romixgen_true = romixgen_true 
        self.romixgen_prob = romixgen_prob
        self.dataset = dataset 
        
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
        
        if (self.romixgen_true) & (random.random() < self.romixgen_prob):
            if self.dataset == 'coco':
                img_id = ann['image_id'].split('_')[-1]
                image,caption, img_id = self.romixgen(img_id)
                return image, caption, self.img_ids['coco'+'_'+img_id]
            elif self.dataset == 'flickr':
                
                # img_id = ann['image_id']
                # image,caption, img_id = self.romixgen(img_id)
                # return image, caption, self.img_ids[img_id]
                
                mix_ann = np.random.choice(self.ann)
                img_id = ann['image_id']
                
                image_path = os.path.join(self.image_root,ann['image'])
                image0 = Image.open(image_path).convert('RGB')
                
                image_path = os.path.join(self.image_root,mix_ann['image'])
                image1 = Image.open(image_path).convert('RGB')
                
                #Mixgen
                image = 0.5 * np.array(self.transform.transforms[0](image0)) + 0.5 * np.array(self.transform.transforms[0](image1))
                image = transforms.Compose(self.transform.transforms[1:])(Image.fromarray(image.astype(np.uint8)))
                
                caption0 = pre_caption(ann['caption'],self.max_words)
                caption1 = pre_caption(mix_ann['caption'],self.max_words)
                caption = caption0 + " " + caption1
                
                return image, caption, self.img_ids[img_id]
            
        else:
            image_path = os.path.join(self.image_root,ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            caption = pre_caption(ann['caption'],self.max_words)
            return image, caption, self.img_ids[ann['image_id']]

class re_mixgen(Dataset):
    def __init__(self, ann_file: str, transform: list, image_root: str, romixgen: object, 
                romixgen_true: bool = True ,romixgen_prob: float = 0.1, max_words: int = 30, dataset: str = 'coco'):        
        '''
        ann_file : annotation file : [{'caption': 'A woman wearing a net on her head cutting a cake. ',
                                        'image ': 'COCO_val2014_000000522418.jpg',
                                        'image_id': 'coco_522418'},...
        
        '''
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
            
        self.transform = transform # transform for nomral image 
        self.image_root = image_root 
        self.max_words = max_words
        
        self.romixgen = romixgen 
        self.romixgen_true = romixgen_true 
        self.romixgen_prob = romixgen_prob
        self.dataset = dataset 
        
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
        mix_ann = np.random.choice(self.ann)
        
        if (self.romixgen_true) & (random.random() < self.romixgen_prob):
            
            if self.dataset == 'coco':
                img_id = ann['image_id'].split('_')[-1]
                image,caption, img_id = self.romixgen(img_id)
                return image, caption, self.img_ids['coco'+'_'+img_id]
            
            elif self.dataset == 'flickr':
                
                image_path = os.path.join(self.image_root,ann['image'])
                image0 = Image.open(image_path).convert('RGB')
                caption0 = pre_caption(ann['caption'],self.max_words)
                
                mix_ann = np.random.choice(self.ann)
                
                image_path = os.path.join(self.image_root,mix_ann['image'])
                image1 = Image.open(image_path).convert('RGB')
                caption1 = pre_caption(mix_ann['caption'],self.max_words)
                
                image = 0.5 * image0 + 0.5 * image1
                caption = caption0 + " " + caption1
                
                return image, caption, self.img_ids[img_id]
            
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
    
class re_eval_perturb_dataset(Dataset):
    def __init__(self, ann_file, img_size, image_root, pertur=None, txt_pertur=None, max_words=30):        
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
    def __init__(self, ann_file, image_root, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root 
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        img_dir = os.path.join(self.image_root, ann['image'])
        image = Image.open(img_dir).convert('RGB')   
        image = self.transform(image)
        
        return image, caption
            

    
