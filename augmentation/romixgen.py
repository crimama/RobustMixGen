import os
import math
import numpy as np
from PIL import Image 
import cv2 
from dataset.utils import pre_caption
import os
from transformers import MarianMTModel, MarianTokenizer
import random 
import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

class MiX:
    def __init__(self, image_dict, obj_bg_dict, img_aug_function, txt_aug_function, 
                normal_image_root, normal_transform):
        self.img_aug = img_aug_function
        self.txt_aug = txt_aug_function 

        self.image_dict = image_dict 
        self.obj_bg_dict = obj_bg_dict

        self.normal_transform = normal_transform 
        self.normal_image_root = normal_image_root 
    
    def normal_load(self,ann):
        image_path = os.path.join(self.normal_image_root,ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.normal_transform(image)
        caption = pre_caption(ann['caption'],50) #50 = max_words
        return image,caption 

    def mix(self,obj_id,bg_id):
        img = self.img_aug(obj_id,bg_id)
        txt = self.txt_aug(obj_id,bg_id)
        return img,txt 
    
    def select_id(self,image_id,obj_bg):
        if obj_bg =='obj':
            obj_id = image_id 
            bg_id = random.choice(self.obj_bg_dict["bg"])
        elif obj_bg == 'bg': 
            bg_id = image_id 
            obj_id = random.choice(self.obj_bg_dict["obj"])            
            
        return obj_id, bg_id 

    def __call__(self,ann): # ann  = image_caption[index] 
        image_id = ann['image_id'].split('_')[-1]
        self.image_id = image_id 
        
        #Typeerror : #obj 혹은 bg에 bbox annotation 정보가 없는 경우 
        #ValueError : #objet image의 크기가 너무 커서 resize해도 bg로 들어가지 않는 경우 
        #UnboundLocalError: #obj 또는 bg 에 unusable_bg가 걸리는 경우 
        
        try:
            obj_id,bg_id = self.select_id(image_id,self.image_dict[image_id]['obj_bg'])
            img,caption = self.mix(obj_id,bg_id)
            
        except  (TypeError, ValueError, UnboundLocalError) : 
            try:
                obj_id,bg_id = self.select_id(image_id,self.image_dict[image_id]['obj_bg'])
                img,caption = self.mix(obj_id,bg_id) 
                
            except  (TypeError, ValueError, UnboundLocalError) : 
                try:
                    obj_id,bg_id = self.select_id(image_id,self.image_dict[image_id]['obj_bg'])
                    img,caption = self.mix(obj_id,bg_id)           
                    
                except  (TypeError, ValueError, UnboundLocalError) : 
                    img,caption = self.normal_load(ann) #self.ann 에 들어있는 ann은 caption이 한개씩 있어서 random choice 하지 않아도 됨 
                    
                    #caption = np.random.choice(caption,1)[0]
        return img,caption
            
            

class RoMixGen_Img:
    def __init__(self,image_dict,image_root,transform_after_mix,resize_ratio=1):
        # dataset json  
        self.image_dict = image_dict 
        
        # Image 
        self.image_root = image_root                   # preprocessed image for augmentation root 
        self.transform_after_mix = transform_after_mix # transforms functions after augmentation 
        self.resize_ratio = resize_ratio               # how large obj image resized 
        
    def bbox_point(self,bboxes):
        y_up = int(bboxes[1])
        y_down = y_up + int(bboxes[3])
        x_left = int(bboxes[0])
        x_right = x_left + int(bboxes[2])
        return x_left,x_right,y_up,y_down
    
    def __cut_obj__(self,bboxes,obj_img):
        x_left,x_right,y_up,y_down = self.bbox_point(bboxes)
        obj_img = np.array(obj_img)[y_up:y_down,x_left:x_right,:]
        return obj_img 
        
    def __obj_pre__(self,obj_img):
        #! obj img 
        # obj 이미지 cutting 
        bboxes = self.obj_inform['max_obj_bbox'] # obj image 정보 중 max obj bbox 가져 옴 
        obj_img = self.__cut_obj__(bboxes,obj_img) # bbox 정보로 이미지 cut 
        
        #배경 이미지의 bbox 포인트 값 및 width, height 계산 
        bg_bboxes = self.bg_inform['max_obj_bbox']
        bg_x_left,bg_x_right,bg_y_up,bg_y_down = self.bbox_point(bg_bboxes)

        width = bg_x_right - bg_x_left
        height = bg_y_down - bg_y_up 

        #resize 기준 및 length 계산 
        length = {'w_h':'height','length':height} if obj_img.shape[0] < obj_img.shape[1] else {'w_h':'width','length':width}
        
        #resize 기준 대비 다른 길이 비율 계산 
        obj_ratio = obj_img.shape[0] / obj_img.shape[1] #height / width 
        #resize 
        if length['w_h'] == 'height': # height 
            dsize = (int(length['length']/obj_ratio*self.resize_ratio),int(length['length']*self.resize_ratio))
            obj_img = cv2.resize(obj_img,dsize=(dsize))
        else:                         # width 
            dsize = (int(length['length']*self.resize_ratio),int(length['length']*obj_ratio*self.resize_ratio))
            obj_img = cv2.resize(obj_img,dsize=(dsize))
        return obj_img 
        
    def __bg_pre__(self,obj_img,bg_img):
        bg_img = np.array(bg_img)
        
        # boj 이미지의 width, height 계산 
        height = obj_img.shape[0]
        width = obj_img.shape[1]
        
        # 배경 이미지의 mid point랑 obj 이미지의 shape으로 붙일 영역 계산
        mid_x = int(self.bg_inform['max_obj_midpoint'][0])
        mid_y = int(self.bg_inform['max_obj_midpoint'][1])

        x_left = mid_x - math.ceil(width/2)
        x_right = mid_x + int(width/2)

        y_up = mid_y - math.ceil(height/2)
        y_down = mid_y + int(height/2)
        
                
        # 배경 벗어나는 것 보정 
        if y_up <0:
            y_down = y_down - y_up
            y_up = y_up - y_up

        if y_down > bg_img.shape[0]:
            move_length = y_down - bg_img.shape[0]
            y_up,y_down = y_up - move_length, y_down - move_length  
            
        if x_left < 0:
            x_left,x_right = x_right-x_left,x_left-x_left
            
        if x_right > bg_img.shape[1]:
            move_length = x_right - bg_img.shape[1]
            x_left,x_right = x_left-move_length,x_right-move_length

        # 배경에 obj 이미지 paste 
        bg_img[y_up:y_down,x_left:x_right,:] = 0 
        bg_img[y_up:y_down,x_left:x_right,:] = obj_img
        
        return bg_img 
        
    def __call__(self,obj_id,bg_id):
        self.obj_inform = self.image_dict[obj_id] # 이미지 전처리 정보 중 해당 obj의 정보를 가져 옴 
        self.bg_inform  = self.image_dict[bg_id]  # 이미지 전처리 정보 중 해당 bg의 정보를 가져 옴 
        
        # image open 
        obj_img = Image.open(os.path.join(self.image_root,'obj',self.obj_inform['file_name'])).convert('RGB')
        bg_img  = Image.open(os.path.join(self.image_root,'bg',self.bg_inform['file_name'])).convert('RGB')
        
        # Preprocess for obj,bg image 
        obj_img = self.__obj_pre__(obj_img)
        bg_img  = self.__bg_pre__(obj_img,bg_img)
        
        # transforms after mix 

        img = self.transform_after_mix(Image.fromarray(bg_img))
        return img 
    
class RoMixGen_Txt:
    def __init__(self, image_caption, first_model_name = 'Helsinki-NLP/opus-mt-en-fr',second_model_name = 'Helsinki-NLP/opus-mt-fr-en'):
        
        # self.first_model_tokenizer  = MarianTokenizer.from_pretrained(first_model_name)
        # self.first_model            = MarianMTModel.from_pretrained(first_model_name)
        # self.second_model_tokenizer = MarianTokenizer.from_pretrained(second_model_name)
        # self.second_model           = MarianMTModel.from_pretrained(second_model_name)

        self.image_caption          = image_caption
        
        
    # def back_translate(self,text):
    #     first_formed_text           = f">>fr<< {text}"
    #     first_translated            = self.first_model.generate(**self.first_model_tokenizer(first_formed_text, return_tensors="pt", padding=True))
    #     first_translated_text       = self.first_model_tokenizer.decode(first_translated[0], skip_special_tokens=True)
    #     second_formed_text          = f">>en<< {first_translated_text}"
    #     second_translated           = self.second_model.generate(**self.second_model_tokenizer(second_formed_text, return_tensors="pt", padding=True))
    #     second_translated_text      = self.second_model_tokenizer.decode(second_translated[0], skip_special_tokens=True)
        
    #     return second_translated_text
    
    def replace_word(self,captions,bg_cats,obj_cats):
        caption = np.random.choice(captions,1)[0]
        '''
        replaced = False 
        for bg_cat, obj_cat in zip(bg_cats,obj_cats):
            if bg_cat in caption:
                caption = caption.replace(bg_cat, obj_cat)
                print('True')
                replaced = True 
                break
        if not replaced:
            caption = random.choice(obj_cats) + " " + caption
        '''
        try:
            (bg_cat_id, bg_cat) = list(filter(lambda x : x[1] in caption.lower(), enumerate(bg_cats)))[0]
            caption = caption.lower().replace(bg_cat,obj_cats[bg_cat_id]).capitalize()
        except IndexError:
            caption = random.choice(obj_cats) + " " + caption
        return caption 

    def __call__(self,obj_id,bg_id):
        
        obj_cat = self.image_caption[obj_id]["max_obj_cat"] + self.image_caption[obj_id]["max_obj_super_cat"]
        bg_cat = self.image_caption[bg_id]["max_obj_cat"] + self.image_caption[bg_id]["max_obj_super_cat"]
        bg_caption = self.image_caption[bg_id]["captions"]
        self.bg_cat = bg_cat
        self.bg_caption = bg_caption 
        self.obj_cat = obj_cat 
        new_caption = self.replace_word(bg_caption, bg_cat, obj_cat)
        
        """obj_tok = self.first_model_tokenizer.encode(obj_caption) # caption to token 
        bg_tok = self.first_model_tokenizer.encode(bg_caption)   # caption to token 
        
        concat_token = obj_tok[:2] + bg_tok[2:] # concat two caption naively 
        concat_text = self.first_model_tokenizer.decode(concat_token, skip_special_tokens=True) # token to caption """
        #backtranslated_text = self.back_translate(new_caption) # Eng - Fren - Eng
        
        #return backtranslated_text # return all 5 captions, 
        return new_caption
                    
                    
class BackTranslation:
    def __init__(self,device,first_model_name = 'Helsinki-NLP/opus-mt-en-fr',second_model_name = 'Helsinki-NLP/opus-mt-fr-en'):
        self.first_model_tokenizer  = MarianTokenizer.from_pretrained(first_model_name)
        self.first_model            = MarianMTModel.from_pretrained(first_model_name).to(device)
        self.second_model_tokenizer = MarianTokenizer.from_pretrained(second_model_name)
        self.second_model           = MarianMTModel.from_pretrained(second_model_name).to(device)
        self.device = device
        
    def __call__(self,text):
        first_formed_text           = [f">>fr<< {text}" for text in list(text)]
        first_token                 = self.first_model_tokenizer(first_formed_text, return_tensors="pt", padding=True).to(self.device)
        first_translated            = self.first_model.generate(**first_token)
        first_translated_text       = self.first_model_tokenizer.batch_decode(first_translated, skip_special_tokens=True)
        second_formed_text          = [f">>en<< {text}" for text in list(first_translated_text)]
        second_token                = self.second_model_tokenizer(second_formed_text, return_tensors="pt", padding=True).to(self.device)
        second_translated           = self.second_model.generate(**second_token)
        second_translated_text      = self.second_model_tokenizer.batch_decode(second_translated, skip_special_tokens=True)
        return second_translated_text                
