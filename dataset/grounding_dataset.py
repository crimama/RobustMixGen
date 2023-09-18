import json
import os
import random 
import numpy as np 

from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption, pertur_check
from torchvision import transforms 
from dataset.randaugment import RandomAugment


class grounding_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root,
                romixgen: object = False, romixgen_true: bool = False, romixgen_prob: float = 0.5,
                max_words=30, mode='train'):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode
        
        self.romixgen = romixgen 
        self.romixgen_true = romixgen_true 
        self.romixgen_prob = romixgen_prob
        
        if self.mode == 'train':
            self.img_ids = {} 
            n = 0
            for ann in self.ann:
                img_id = ann['image'].split('/')[-1]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1                    

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        ann = self.ann[index]
        
        img_dir = ann['image'].split('/')[-1]
        img_id  = img_dir.split('_')[-1].lstrip('0').split('.jpg')[0]
        if (self.romixgen_true) & (random.random() < self.romixgen_prob):
            image,caption, img_id = self.romixgen(img_id)

        else:
            image_path = os.path.join(self.image_root,ann['image'])            
            image = Image.open(image_path).convert('RGB')  
            image = self.transform(image)
            caption = pre_caption(ann['text'], self.max_words) 
        
        if self.mode=='train':
            return image, caption, self.img_ids[ann['image'].split('/')[-1]]
        else:
            return image, caption, ann['ref_id']
        
class grounding_pertur_dataset(grounding_dataset):
    def __init__(self, ann_file, img_res, image_root, mode='test',
                    img_pertur=None, txt_pertur=None):        
        super(grounding_pertur_dataset, self).__init__(
                                                        ann_file   = ann_file,
                                                        transform  = self.get_transform(img_res),
                                                        image_root = image_root,
                                                        mode       = mode 
                                                    )
        
        self.img_pertur = pertur_check(img_pertur)
        self.txt_pertur = pertur_check(txt_pertur)
        
    def get_transform(self, img_size):
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        test_transform      = transforms.Compose([
                                                transforms.Resize((img_size,img_size),interpolation=Image.BICUBIC),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])
        return test_transform
    
    def __getitem__(self, index):
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')   
        image = self.img_pertur(np.array(image),severity=2).astype(np.uint8)
        image = self.transform(Image.fromarray(image).convert('RGB'))     
        
        caption = pre_caption(ann['text'], self.max_words)
        caption = self.txt_pertur(caption)
        
        return image, caption, ann['ref_id']