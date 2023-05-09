import os
import math
import numpy as np
from PIL import Image 
import cv2 
from dataset.utils import pre_caption
import os
from transformers import MarianMTModel, MarianTokenizer
import random 
import json


class MiX:
    def __init__(self, img_aug_function, txt_aug_function,
                normal_image_root, normal_transform,
                image_info, obj_bg_threshold = 0.01, bg_center_threshold = 0.7):
        
        self.img_aug = img_aug_function
        self.txt_aug = txt_aug_function 
        self.normal_transform = normal_transform 
        self.normal_image_root = normal_image_root

        self.image_info_dict, self.obj_bg_dict = self.romixgen_preset(self.use_sub_dict_key(json.load(open(image_info,'r'),),
                                                    ['file_name', 'width', 'height', 'max_obj_super_cat','max_obj_cat','max_obj_area', 'max_obj_midpoint','max_obj_bbox', 'max_obj_area_portion','captions']),
                                                        obj_bg_threshold, bg_center_threshold) 
            
        #after initializing img_infodict and obj_bg_dict, pass them to romixgen_img, and romixgen_txt

        self.img_aug.__get_dict__(self.use_sub_dict_key(self.image_info_dict,['file_name', 'max_obj_midpoint', 'max_obj_bbox']))
        self.txt_aug.__get_dict__(self.use_sub_dict_key(self.image_info_dict,['file_name', 'max_obj_super_cat', 'max_obj_cat','captions']))

    #json 파일에서 필요한 key만 뽑아서 사용
    def use_sub_dict_key(self, dic: dict, key_to_use: list):
        return_dic = {}
        for item in dic:
            for sub_item in dic[item]:
                if sub_item in key_to_use:
                    return_dic[item] = return_dic.get(item, {})
                    return_dic[item][sub_item] = dic[item][sub_item]
                
        return return_dic

    # bg_center_threshold 이용하는 함수
    def center_check(self, midpoint:list, width:int ,height:int ,thrs: float):
        width_area = [0+width*((1-thrs)/2), width-width*((1-thrs)/2)]
        height_area = [0+height*((1-thrs)/2), height-height*((1-thrs)/2)]
        try:
            if midpoint[0] > width_area[0] and midpoint[0] < width_area[1]:
                if midpoint[1] > height_area[0] and midpoint[1] < height_area[1]:
                    return True
                else:
                    return False
            else:
                return False
        except:
            return False
    
    # romixgen 적용한 dictionary를 만들어주는 함수
    def romixgen_preset(self, img_info_dict, obj_bg_threshold = 0.01, bg_center_threshold = 0.7): 
        seg_or_bbox = 'bbox'
        for key in (img_info_dict.keys()):
            try:
                max_obj_area_portion = img_info_dict[key]['max_obj_area_portion']
                max_obj_midpoint = img_info_dict[key]['max_obj_midpoint']
                img_width, img_height = int(img_info_dict[key]['width']), int(img_info_dict[key]['height'])

                if max_obj_area_portion:
                    img_info_dict[key].pop('max_obj_segment_points', None) if seg_or_bbox == 'bbox' else img_info_dict[key].pop('max_obj_bbox', None)

                    if max_obj_area_portion > obj_bg_threshold: # 물체가 이미지의 일정 비율 이상 차지하는 경우
                        img_info_dict[key]["obj_bg"] = "obj"
                    else:
                        if self.center_check(max_obj_midpoint, img_width, img_height, bg_center_threshold): # 빈 곳의 중심이 이미지의 중심에 가까운 경우
                            img_info_dict[key]["obj_bg"] = "bg"
                        else: # 빈 곳의 중심이 이미지의 중심에 가깝지 않은 경우 (외곽에 위치한 경우)
                            img_info_dict[key]["obj_bg"] = "unusable_bg"

                else:  # max obj 비어있는 경우 (그냥 real bg로 저장)
                    img_info_dict[key]["obj_bg"] = "unusable_bg"

            except Exception as e:
                print(f"Error processing image {img_info_dict[key]['file_name']}: {e}")
        obj_bg_dict={}

        #obj,bg key를 가진 dictionary 생성
        for key in img_info_dict.keys():
            if img_info_dict[key]["obj_bg"] not in obj_bg_dict:
                obj_bg_dict[img_info_dict[key]["obj_bg"]] = [key]
            else:
                obj_bg_dict[img_info_dict[key]["obj_bg"]].append(key)

        return img_info_dict, obj_bg_dict

                
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
            obj_id,bg_id = self.select_id(image_id,self.image_info_dict[image_id]['obj_bg'])
            img,caption = self.mix(obj_id,bg_id)
            
        except  (TypeError, ValueError, UnboundLocalError) : 
            try:
                obj_id,bg_id = self.select_id(image_id,self.image_info_dict[image_id]['obj_bg'])
                img,caption = self.mix(obj_id,bg_id) 
                
            except  (TypeError, ValueError, UnboundLocalError) : 
                try:
                    obj_id,bg_id = self.select_id(image_id,self.image_info_dict[image_id]['obj_bg'])
                    img,caption = self.mix(obj_id,bg_id)           
                    
                except  (TypeError, ValueError, UnboundLocalError) : 
                    img,caption = self.normal_load(ann) #self.ann 에 들어있는 ann은 caption이 한개씩 있어서 random choice 하지 않아도 됨 
                    
                    #caption = np.random.choice(caption,1)[0]
        return img,caption

                
class RoMixGen_Img:
    def __init__(self, image_root, transform_after_mix, midset_bool, resize_ratio=1 ):
        # Image 
        self.image_root = image_root                   # preprocessed image for augmentation root 
        self.transform_after_mix = transform_after_mix # transforms functions after augmentation 
        self.midset_bool = midset_bool                # if True, use midset
        self.resize_ratio = resize_ratio                # how large obj image resized 

    def __get_dict__(self,img_info):
        self.img_info_dict = img_info

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
        # obj 이미지 cutting 
        bboxes = self.obj_inform['max_obj_bbox'] # obj image 정보 중 max obj bbox 가져 옴 
        obj_img = self.__cut_obj__(bboxes,obj_img) # bbox 정보로 이미지 cut 
        if self.midset_bool:
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
    
        # obj 이미지의 width, height 계산 
        height = obj_img.shape[0]
        width = obj_img.shape[1]
        
        if self.midset_bool:
            # 배경 이미지의 mid point랑 obj 이미지의 shape으로 붙일 영역 계산
            mid_x = int(self.bg_inform['max_obj_midpoint'][0])
            mid_y = int(self.bg_inform['max_obj_midpoint'][1])
        else:
            mid_x = int(self.obj_inform['max_obj_midpoint'][0])
            mid_y = int(self.obj_inform['max_obj_midpoint'][1])

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
        
        self.obj_inform = self.img_info_dict[obj_id] # 이미지 전처리 정보 중 해당 obj의 정보를 가져 옴 
        self.bg_inform  = self.img_info_dict[bg_id]  # 이미지 전처리 정보 중 해당 bg의 정보를 가져 옴 
        
        # image open 
        obj_img = Image.open(os.path.join(self.image_root, self.obj_inform['file_name'])).convert('RGB')
        bg_img  = Image.open(os.path.join(self.image_root, self.bg_inform['file_name'])).convert('RGB')
        
        # Preprocess for obj,bg image 
        obj_img = self.__obj_pre__(obj_img)
        bg_img  = self.__bg_pre__(obj_img,bg_img)
        
        # transforms after mix 
        img = self.transform_after_mix(Image.fromarray(bg_img))

        return img 
    
    
class RoMixGen_Txt:
    def __init__(self, first_model_name = 'Helsinki-NLP/opus-mt-en-fr',second_model_name = 'Helsinki-NLP/opus-mt-fr-en'):
        self.first_model_tokenizer  = MarianTokenizer.from_pretrained(first_model_name)
        self.first_model            = MarianMTModel.from_pretrained(first_model_name)
        self.second_model_tokenizer = MarianTokenizer.from_pretrained(second_model_name)
        self.second_model           = MarianMTModel.from_pretrained(second_model_name)

        #self.image_caption          = image_caption

    def __get_dict__(self,img_info):
        self.img_info_dict = img_info
    
    def back_translate(self,text):
        first_formed_text           = f">>fr<< {text}"
        first_translated            = self.first_model.generate(**self.first_model_tokenizer(first_formed_text, return_tensors="pt", padding=True))
        first_translated_text       = self.first_model_tokenizer.decode(first_translated[0], skip_special_tokens=True)
        second_formed_text          = f">>en<< {first_translated_text}"
        second_translated           = self.second_model.generate(**self.second_model_tokenizer(second_formed_text, return_tensors="pt", padding=True))
        second_translated_text      = self.second_model_tokenizer.decode(second_translated[0], skip_special_tokens=True)
        return second_translated_text
    
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
        
        obj_cat = self.img_info_dict[obj_id]["max_obj_cat"] + self.img_info_dict[obj_id]["max_obj_super_cat"]
        bg_cat = self.img_info_dict[bg_id]["max_obj_cat"] + self.img_info_dict[bg_id]["max_obj_super_cat"]
        bg_caption = self.img_info_dict[bg_id]["captions"]
        new_caption = self.replace_word(bg_caption, bg_cat, obj_cat)
        
        """obj_tok = self.first_model_tokenizer.encode(obj_caption) # caption to token 
        bg_tok = self.first_model_tokenizer.encode(bg_caption)   # caption to token 
        
        concat_token = obj_tok[:2] + bg_tok[2:] # concat two caption naively 
        concat_text = self.first_model_tokenizer.decode(concat_token, skip_special_tokens=True) # token to caption """
        backtranslated_text = self.back_translate(new_caption) # Eng - Fren - Eng
        #backtranslated_text = self.back_translate(concat_text) # Eng - Fren - Eng 
        
        return backtranslated_text # return all 5 captions, 
        #return new_caption
                    
                    
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