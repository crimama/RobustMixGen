import os 
from PIL import Image 
import json 
import yaml 
import pandas as pd 

from torchvision import transforms 

from augmentation.romixgen import RoMixGen_Img, RoMixGen_Txt,MiX


def create_romixgen(config):
    
    transform_after_mix = transforms.Compose([
                                            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                        ])
    
    
    image_dict         = json.load(open(config['image_dict_file']))
    obj_bg_dict        = json.load(open(config['obj_bg_dict_file']))
    
    
    img_func = RoMixGen_Img(image_dict            = image_dict,
                            image_root            = config['aug_image_root'],
                            transform_after_mix   = transform_after_mix,
                            resize_ratio          = config['romixgen_resize_ratio'])

    txt_func = RoMixGen_Txt(image_caption       = image_dict)
         
    romixgen = MiX( image_dict        = image_dict,
                    obj_bg_dict       = obj_bg_dict,
                    img_aug_function  = img_func,
                    txt_aug_function  = txt_func,
                    normal_image_root = config['image_root'],
                    normal_transform  = transform_after_mix)
    
    return romixgen