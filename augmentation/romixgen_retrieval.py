import os 
import random 
import json 
import pandas as pd 
from .txt_aug import txt_aug_function 
from .img_aug import img_aug_function
from .base_romixgen import BaseRomixgen

class RetrievalRomixgen(BaseRomixgen):
    def __init__(self, image_info_dir: str, image_root:str, transform, image_mix_ratio:float,
                    txt_method:str, txt_pertur:bool, obj_bg_threshold:float):
        super(RetrievalRomixgen, self).__init__(
                                                image_info_dir   = image_info_dir, 
                                                image_root       = image_root, 
                                                transform        = transform, 
                                                image_mix_ratio  = image_mix_ratio, 
                                                txt_method       = txt_method,
                                                txt_pertur       = txt_pertur, 
                                                obj_bg_threshold = obj_bg_threshold
                                                )
    def get_image_info(self, image_info_dir, obj_bg_threshold):
        '''
        Image info 파일 load 하면서 각 obj, bg로 분류 
        '''
        
        if type(obj_bg_threshold) != list:
            obj_bg_threshold = [obj_bg_threshold, 1.0]
            
        image_info = json.load(open(image_info_dir))
        for key in image_info.keys():
            # BBOX 영역 비율 계산 
            if 'max_obj_bbox' in  image_info[key].keys():
                if image_info[key]['max_obj_bbox']:
                    img_width, img_height = int(image_info[key]["width"]), int(image_info[key]["height"])
                    max_obj_area_portion = cal_area_portion(image_info[key]['max_obj_bbox'],img_width, img_height)
                    image_info[key]['mop'] = max_obj_area_portion
                    image_info[key]["obj_bg"] = 'obj' if (obj_bg_threshold[0] <= max_obj_area_portion <= obj_bg_threshold[1]) else 'bg'
                else:
                    image_info[key]['mop'] = 0 
                    image_info[key]['obj_bg'] = 'Unusuable'
            else:
                image_info[key]['mop'] = 0 
                image_info[key]['obj_bg'] = 'Unusuable'
                
        return image_info     
    
    
    def mix(self, obj_id, bg_id):
        obj_info, bg_info = self.image_info[obj_id], self.image_info[bg_id]
        img = self.img_aug(obj_info, bg_info)
        txt = self.txt_aug(obj_info['captions'], bg_info['captions'])
        return img, txt 
    
    def __call__(self, image_id: str):
        self.image_id = image_id 
        obj_id, bg_id, image_id = self.select_id(image_id)
        img, caption = self.mix(obj_id, bg_id)
        return img, caption, image_id  


class mixgen(BaseRomixgen):
    def __init__(self, image_info_dir: str, image_root:str, transform, image_mix_ratio:float,
                    txt_method:str, txt_pertur:bool, obj_bg_threshold:float):
        super(mixgen, self).__init__(
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
        img = self.img_aug(obj_info, bg_info)
        txt = self.txt_aug(obj_info['captions'], bg_info['captions'])
        return img, txt 
    
    def __call__(self, image_id: str):
        self.image_id = image_id 
        obj_id, bg_id, image_id = self.select_id(image_id)
        img, caption = self.mix(obj_id, bg_id)
        return img, caption, image_id  
    
def cal_area_portion(bbox, width, height):
    X,Y,W,H = bbox
    area_portion = (W * H) / (width * height)
    return area_portion 