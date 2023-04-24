import os
import math
import numpy as np
from PIL import Image 
import cv2 
from dataset.utils import pre_caption
import os
from transformers import MarianMTModel, MarianTokenizer
import random 

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

    def __call__(self,ann): # ann  = image_caption[index] 
        image_id = ann['image_id'].split('_')[-1]
        try:
            if self.image_dict[image_id]['obj_bg'] =='obj':
                obj_id = image_id 
                bg_id = random.choice(self.obj_bg_dict["bg"])
                img,caption = self.mix(obj_id,bg_id) 
                
            elif self.image_dict[image_id]['obj_bg'] == 'bg':
                bg_id = image_id 
                obj_id = random.choice(self.obj_bg_dict["obj"])
                img,caption = self.mix(obj_id,bg_id)      
                      
            else:
                img,caption = self.normal_load(ann)
        except:
            img,caption = self.normal_load(ann) 

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
    def __init__(self, image_caption, first_model_name='Helsinki-NLP/opus-mt-en-fr', second_model_name='Helsinki-NLP/opus-mt-fr-en'):
        self.first_model_tokenizer  = MarianTokenizer.from_pretrained(first_model_name)
        self.first_model            = MarianMTModel.from_pretrained(first_model_name)
        self.second_model_tokenizer = MarianTokenizer.from_pretrained(second_model_name)
        self.second_model           = MarianMTModel.from_pretrained(second_model_name)
        self.image_caption          = image_caption
        self.translated_text_cache  = {}

    def back_translate(self, text):
        if text in self.translated_text_cache:
            return self.translated_text_cache[text]

        first_formed_text        = f">>fr<< {text}"
        first_translated         = self.first_model.generate(**self.first_model_tokenizer(first_formed_text, return_tensors="pt", padding=True))
        first_translated_text    = self.first_model_tokenizer.decode(first_translated[0], skip_special_tokens=True)
        second_formed_text       = f">>en<< {first_translated_text}"
        second_translated        = self.second_model.generate(**self.second_model_tokenizer(second_formed_text, return_tensors="pt", padding=True))
        second_translated_text   = self.second_model_tokenizer.decode(second_translated[0], skip_special_tokens=True)
        
        self.translated_text_cache[text] = second_translated_text
        return second_translated_text

    def replace_word(self, captions, bg_cats, obj_cats):
        replaced = False
        for bg_cat, obj_cat in zip(bg_cats, obj_cats):
            if bg_cat in captions:
                captions = captions.replace(bg_cat, obj_cat)
                replaced = True
        if not replaced:
            captions = random.choice(obj_cats) + " " + captions
        return captions

    def __call__(self,obj_id,bg_id):
        
        obj_cat = self.image_caption[obj_id]["max_obj_cat"] + self.image_caption[obj_id]["max_obj_super_cat"]
        bg_cat = self.image_caption[bg_id]["max_obj_cat"] + self.image_caption[bg_id]["max_obj_super_cat"]

        bg_caption = self.image_caption[bg_id]["captions"]

        new_caption = [self.replace_word(bg_caption_item, bg_cat, obj_cat) for bg_caption_item in bg_caption]
        backtranslated_text = [self.back_translate(new_caption_item) for new_caption_item in new_caption]
        #backtranslated_text = self.back_translate(new_caption) 
        
        return backtranslated_text # return all 5 captions, 
        
                    