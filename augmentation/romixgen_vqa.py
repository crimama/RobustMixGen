import random 
import json 
from pathlib import Path
import numpy as np 
import pandas as pd 
from augmentation.base_romixgen import cal_area_portion, BaseRomixgen
from dataset.vqa_dataset import cal_answer_weights
from .txt_aug import txt_aug_function 
from .img_aug import img_aug_function

class VQARomixgen(BaseRomixgen):
    def __init__(self, vg_info_dir:Path, vqa_info_dir:Path,  vqa_root:Path, vg_root:Path, transform, image_mix_ratio:float,
                    txt_method:str, txt_pertur:bool, txt_eos:str, vg_obj_bg_threshold:float, vqa_obj_bg_threshold:float):
                
        self.vqa_img_aug = img_aug_function(vqa_root, transform, image_mix_ratio)
        self.vg_img_aug  = img_aug_function(vg_root, transform, image_mix_ratio)
        self.txt_aug = txt_aug_function(txt_method, txt_pertur)
        self.txt_eos = txt_eos 
        
        self.vg_info  = self.get_image_info(vg_info_dir, vg_obj_bg_threshold)
        self.vqa_info  = self.get_image_info(vqa_info_dir, vqa_obj_bg_threshold)
        self.obj_bg_vg_pool = self.get_obj_bg_pool(self.vg_info,'vg')    
        self.obj_bg_vqa_pool = self.get_obj_bg_pool(self.vqa_info)    
        
    def get_obj_bg_pool(self, image_info, task='vqa'):
            '''
            앞서 만든 Image info 를 이용해 obj,bg를 key로, image id들을 value로 사용하는 dict 생성 
            '''
            obj_bg = [] 
            for key in image_info.keys():
                obj_bg.append([image_info[key]['obj_bg'], image_info[key]['file_name']])
                
            obj_bg = pd.DataFrame(obj_bg)    
            if task == 'vqa':
                obj_bg[1] = obj_bg[1].apply(lambda x : x.split('_')[-1].lstrip('0').split('.jpg')[0]) #Image id 전처리 
            elif task == 'vg':
                obj_bg[1] = obj_bg[1].apply(lambda x : x.split('.jpg')[0]) # obj_bg[1] : file_name column
            get_obj_bg_pool = {
                                'obj': list(obj_bg[obj_bg[0] == 'obj'][1].values),
                                'bg' : list(obj_bg[obj_bg[0] == 'bg'][1].values),
                                }
            return get_obj_bg_pool

    def select_id(self, ann:dict) -> str:
        '''
        unusuable의 경우 기존 이미지가 사용되지 않기 때문에 object id가 이를 대체 
        '''
        if ann['dataset'] == 'vqa':
            image_id = ann['image'].split('_')[-1].split('.jpg')[0].lstrip('0')
            obj_bg = self.vqa_info[image_id]['obj_bg']
        elif ann['dataset'] == 'vg':
            image_id = ann['image'].split('.jpg')[0]
            obj_bg = self.vg_info[image_id]['obj_bg']        
        else:
            raise 'No obj_bg'
        
        if obj_bg == "obj":
            obj_id = image_id
            if ann['dataset'] == 'vqa':
                bg_id  = random.choice(self.obj_bg_vqa_pool["bg"])
            elif ann['dataset'] == 'vg':
                bg_id  = random.choice(self.obj_bg_vg_pool["bg"])
        elif obj_bg == "bg":
            bg_id  = image_id 
            if ann['dataset'] == 'vqa':
                obj_id  = random.choice(self.obj_bg_vqa_pool["obj"])
            elif ann['dataset'] == 'vg':
                obj_id  = random.choice(self.obj_bg_vg_pool["obj"])
            
        elif obj_bg == 'Unusuable':
            if ann['dataset'] == 'vqa':
                obj_id = random.choice(self.obj_bg_vqa_pool["obj"])
                bg_id  = random.choice(self.obj_bg_vqa_pool["bg"])
            elif ann['dataset'] == 'vg':
                obj_id = random.choice(self.obj_bg_vg_pool["obj"])
                bg_id  = random.choice(self.obj_bg_vg_pool["bg"])
            image_id = obj_id 
            
        return obj_id, bg_id, image_id 
        
    def mix(self, obj_id:str, bg_id:str, dataset:str):
        if dataset == 'vqa':
            # Image augmentation 
            obj_info, bg_info = self.vqa_info[obj_id], self.vqa_info[bg_id]            
            image = self.vqa_img_aug(obj_info, bg_info)
            
            # Text augmentation         
            obj_qa = np.random.choice(obj_info['QA-pair'])
            bg_qa = np.random.choice(bg_info['QA-pair'])
            question = self.txt_aug([obj_qa['question']], [bg_qa['question']])
            answers = [self.txt_aug([obj_an], [bg_an]) for obj_an, bg_an in zip(obj_qa['answer'],bg_qa['answer'])]
            
        elif dataset == 'vg':
            # Image augmentation 
            obj_info, bg_info = self.vg_info[obj_id], self.vg_info[bg_id]
            image = self.vg_img_aug(obj_info, bg_info)
            
            # Text augmentation
            obj_qa = np.random.choice(obj_info['QA-pair'])
            bg_qa = np.random.choice(bg_info['QA-pair'])
            question = self.txt_aug([obj_qa['question']], [bg_qa['question']])
            answers = self.txt_aug([obj_qa['answer']], [bg_qa['answer']])
            
        
        answers, weights = cal_answer_weights(answers, dataset, self.txt_eos)
        return image, question, answers, weights 
    
    def __call__(self, ann:dict): 
        self.ann = ann 
        self.dataset = ann['dataset']
        
        obj_id, bg_id, image_id = self.select_id(ann)
        self.obj_id, self.bg_id = obj_id, bg_id 
        image, question, answers, weights  = self.mix(obj_id, bg_id, ann['dataset'])
        return image, question, answers, weights 
        