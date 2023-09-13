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
    # for i in range(num):
    #     # image mixup
    #     image[i,:] = lam * image[i,:] + (1 - lam) * image[i+num,:] # original code 
    #     # text concat
    #     text[i] = text[i] + " " + text[i+num] # original code 
    
    # Image 
    image[:num,:] = lam * image[:num,:] + (1-lam) + image[num:2*num,:]
    
    for i in range(num): 
        # text concat
        text[i] = text[i] + " " + text[i+num] # original code 
    
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
