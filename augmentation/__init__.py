import os 
from PIL import Image 
import json 
import yaml 
import pandas as pd 

from torchvision import transforms 

from augmentation.romixgen import RoMixGen_Img, RoMixGen_Txt,MiX
from dataset.randaugment import RandomAugment

def create_romixgen(config):
    if config['hard_aug']:
        transform_after_mix     = transforms.Compose([                        
                                            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                                            transforms.RandomHorizontalFlip(),
                                            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                                            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                            ])  
    else:
        transform_after_mix = transforms.Compose([
                                            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                        ])
    
    #config             = yaml.load(open('./configs/Retrieval_coco_romix.yaml'),Loader=yaml.Loader)
    
    
    img_func = RoMixGen_Img(image_root =            config['image_root'],
                            transform_after_mix =   transform_after_mix,
                            midset_bool=            config['img_midset'],
                            resize_ratio =          config['mixgen_resize_ratio'])

    txt_func = RoMixGen_Txt()
         
    romixgen = MiX( img_aug_function  = img_func,
                    txt_aug_function  = txt_func,
                    normal_image_root = config['image_root'],
                    normal_transform  = transform_after_mix,
                    image_info        = config['img_info_json'],
                    obj_bg_threshold  = config['obj_bg_threshold'],
                    bg_center_threshold = config['bg_center_threshold'],)
    
    return romixgen

