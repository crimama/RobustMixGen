import os 
import random 
import json 
import pandas as pd 
from .txt_aug import txt_aug_function 
from .img_aug import img_aug_function
from .base_romixgen import BaseRomixgen

class NLVRRomixgen(BaseRomixgen):
    def __init__(self, image_info_dir: str, image_root:str, transform, image_mix_ratio:float,
                    txt_method:str, txt_pertur:bool, obj_bg_threshold:float):
        super(NLVRRomixgen, self).__init__(
                                                image_info_dir   = image_info_dir, 
                                                image_root       = image_root, 
                                                transform        = transform, 
                                                image_mix_ratio  = image_mix_ratio, 
                                                txt_method       = txt_method,
                                                txt_pertur       = txt_pertur, 
                                                obj_bg_threshold = obj_bg_threshold
                                                )
        
    def mix(self, obj_id, bg_id):
        obj_info, bg_info = self.image_info[obj_id], self.image_info[bg_id]
        image = self.img_aug(obj_info, bg_info)
        return image
    
    def __call__(self, ann:dict): 
        image_id0, image_id1 = ann['images']
        obj_id0, bg_id0, image_id0 = self.select_id(image_id0)
        obj_id1, bg_id1, image_id1 = self.select_id(image_id1)
        
        image0 = self.mix(obj_id0, bg_id0)
        image1 = self.mix(obj_id1, bg_id1)
        
        sentence = ann['sentence']
        label = ann['label']
    
        return image0, image1, sentence, label
    