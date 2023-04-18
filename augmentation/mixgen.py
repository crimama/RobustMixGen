"""
MixGen: A New Multi-Modal Data Augmentation
https://arxiv.org/abs/2206.08358
Apache-2.0 License, Copyright 2022 Amazon
"""
import json
import numpy as np
from PIL import Image 
import os 

def mixgen(image, text, num, lam=0.5):
    # default MixGen
    for i in range(num):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[i+num,:]
        # text concat
        text[i] = text[i] + " " + text[i+num]
    return image, text


def mixgen_batch(image, text, num, lam=0.5):
    batch_size = image.size()[0]
    index = np.random.permutation(batch_size)
    for i in range(batch_size):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[index[i],:]
        # text concat
        text[i] = text[i] + " " + text[index[i]]
    return image, 

class MixGen:
    def __init__(self,ann_file,image_root,transform,lam = 0.5):
        self.ann_file   = json.load(open(ann_file,'r'))
        self.image_root = image_root
        self.transform  = transform 
        self.lam  = lam
        
    def __call__(self,ann):
        ann1 = ann
        image1 = Image.open(os.path.join(self.image_root,ann1['image']))
        text1 = ann1['caption']
        
        ann2 = np.random.choice(self.ann_file)
        image2 = Image.open(os.path.join(self.image_root,ann2['image']))
        text2 = ann2['caption']
        
        image = self.lam * self.transform(image1) + (1-self.lam) * self.transform(image2)
        text = text1 + ' ' + text2
        
        return image,text