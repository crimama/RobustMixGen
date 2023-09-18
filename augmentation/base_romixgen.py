import os 
import random 
import json 
import pandas as pd 
from .txt_aug import txt_aug_function 
from .img_aug import img_aug_function

class BaseRomixgen:    
    def __init__(self, image_info_dir: str, image_root:str, transform, image_mix_ratio:float,
                    txt_method:str, txt_pertur:bool, obj_bg_threshold:float):
        
        self.img_aug = img_aug_function(image_root, transform, image_mix_ratio)
        self.txt_aug = txt_aug_function(txt_method, txt_pertur)
        
        self.image_info  = self.get_image_info(image_info_dir, obj_bg_threshold)
        self.obj_bg_pool = self.get_obj_bg_pool(self.image_info)
        
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
                    image_info[key]['obj_bg'] = 'bg'
            else:
                image_info[key]['mop'] = 0 
                image_info[key]['obj_bg'] = 'bg'
                
        return image_info 
    
    def get_obj_bg_pool(self, image_info):
        '''
        앞서 만든 Image info 를 이용해 obj,bg를 key로, image id들을 value로 사용하는 dict 생성 
        '''
        obj_bg = [
            [image_info[key]['obj_bg'], image_info[key]['file_name']] for key in image_info.keys()
        ]
        obj_bg = pd.DataFrame(obj_bg)    
        obj_bg[1] = obj_bg[1].apply(lambda x : x.split('_')[-1].lstrip('0').split('.jpg')[0]) #Image id 전처리 
        return  {
                'obj': list(obj_bg[obj_bg[0] == 'obj'][1].values),
                'bg' : list(obj_bg[obj_bg[0] == 'bg'][1].values),
                }
         
    
    def select_id(self, image_id):
        '''
        image_id 의 경우 retrieval model에서 사용하기 때문에 return 
        unusuable의 경우 기존 이미지가 사용되지 않기 때문에 object id가 이를 대체 
        '''
        obj_bg = self.image_info[image_id]["obj_bg"]
        if obj_bg == "obj":
            obj_id = image_id
            bg_id  = random.choice(self.obj_bg_pool["bg"])
        elif obj_bg == "bg":
            bg_id  = image_id 
            obj_id = random.choice(self.obj_bg_pool["obj"])
        elif obj_bg == 'Unusuable':
            obj_id = random.choice(self.obj_bg_pool["obj"])
            bg_id  = random.choice(self.obj_bg_pool["bg"])
            image_id = obj_id 
        return obj_id, bg_id, image_id 
    
    def mix(self, obj_id, bg_id):
        obj_info, bg_info = self.image_info[obj_id], self.image_info[bg_id]
        img = self.img_aug(obj_info, bg_info)
        txt = self.txt_aug(obj_info['captions'], bg_info['captions'])
        return img, txt 
    
    def __call__(self, image_id):
        self.image_id = image_id 
        obj_id, bg_id, image_id = self.select_id(image_id)
        img, caption = self.mix(obj_id, bg_id)
        return img, caption, image_id  
    
def cal_area_portion(bbox, width, height):
            X,Y,W,H = bbox
            area_portion = (W * H) / (width * height)
            return area_portion 