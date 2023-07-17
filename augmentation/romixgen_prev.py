import json
import os
import random
from itertools import chain    

import cv2
import numpy as np
import torch.distributed as dist
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer
from perturbation.text_perturbation import  get_method_chunk_train

import re 

class MiX:
    def __init__(self, image_info: dict, img_aug_function, txt_aug_function,
                  obj_bg_threshold: float, bg_center_threshold: float, 
                  normal_image_root: str, normal_transform):
        
        self.img_aug = img_aug_function
        self.txt_aug = txt_aug_function 

        self.image_info_dict, self.obj_bg_dict = self.romixgen_preset(
                                                                        self.use_sub_dict_key(json.load( open(image_info, "r"),),["file_name", "width", "height", "max_obj_super_cat","max_obj_cat", "max_obj_area", "max_obj_midpoint", "max_obj_bbox", "max_obj_area_portion", "captions",],),
                                                                        obj_bg_threshold,bg_center_threshold,
                                                                        )
        self.img_aug.__get_dict__(
                                    self.use_sub_dict_key(
                                        self.image_info_dict, ["file_name", "max_obj_midpoint", "max_obj_bbox"]
                                    )
                                )
        self.txt_aug.__get_dict__(
                                    self.use_sub_dict_key(
                                        self.image_info_dict,
                                        ["file_name", "max_obj_super_cat", "max_obj_cat", "captions"],
                                    )
                                )
        self.normal_transform = normal_transform 
        self.normal_image_root = normal_image_root 
        
    # json 파일에서 필요한 key만 뽑아서 사용
    def use_sub_dict_key(self, dic: dict, key_to_use: list):
        return_dic = {}
        for item in dic:
            for sub_item in dic[item]:
                if sub_item in key_to_use:
                    return_dic[item] = return_dic.get(item, {})
                    return_dic[item][sub_item] = dic[item][sub_item]

        return return_dic

    # bg_center_threshold 이용하는 함수
    def center_check(self, midpoint: list, width: int, height: int, thrs: float):
        width_area = [0 + width * ((1 - thrs) / 2), width - width * ((1 - thrs) / 2)]
        height_area = [
            0 + height * ((1 - thrs) / 2),
            height - height * ((1 - thrs) / 2),
        ]
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

    # 각 이미지에 obj / bg 라벨링 
    def romixgen_preset(self, img_info_dict, obj_bg_threshold=0.01, bg_center_threshold=0.7):
        seg_or_bbox = "bbox"
        for key in img_info_dict.keys():
            try:
                #max_obj_area_portion = img_info_dict[key]["max_obj_area_portion"]
                max_obj_midpoint = img_info_dict[key]["max_obj_midpoint"]
                img_width, img_height = int(img_info_dict[key]["width"]), int(img_info_dict[key]["height"])
                try:
                    max_obj_area_portion = (img_info_dict[key]["max_obj_bbox"][2] * img_info_dict[key]["max_obj_bbox"][3])/(img_width * img_height)
                except:
                    max_obj_area_portion = None
                img_info_dict[key]["mop"] = max_obj_area_portion
                if max_obj_area_portion:
                    img_info_dict[key].pop( "max_obj_segment_points", None ) if seg_or_bbox == "bbox" else img_info_dict[key].pop("max_obj_bbox", None)
                    if (max_obj_area_portion > obj_bg_threshold):  # 물체가 이미지의 일정 비율 이상 차지하는 경우
                        img_info_dict[key]["obj_bg"] = "obj"
                    else:
                        if self.center_check( max_obj_midpoint, img_width, img_height, bg_center_threshold):  # 빈 곳의 중심이 이미지의 중심에 가까운 경우
                            img_info_dict[key]["obj_bg"] = "bg"
                        else:  # 빈 곳의 중심이 이미지의 중심에 가깝지 않은 경우 (외곽에 위치한 경우)
                            img_info_dict[key]["obj_bg"] = "unusable_bg"
                else:  # max obj 비어있는 경우 (그냥 unusable bg로 저장)
                    img_info_dict[key]["obj_bg"] = "unusable_bg"
            except Exception as e:
                print(f"Error processing image {img_info_dict[key]['file_name']}: {e}")
        
        # 각 이미지 id 별로 라벨링 된 obj / bg를 모아서 obj_bg_dict 만들어줌 
        obj_bg_dict = {}
        for key in img_info_dict.keys():
            if img_info_dict[key]["obj_bg"] not in obj_bg_dict:
                obj_bg_dict[img_info_dict[key]["obj_bg"]] = [key]
            else:
                obj_bg_dict[img_info_dict[key]["obj_bg"]].append(key)

        return img_info_dict, obj_bg_dict

    def mix(self,obj_id,bg_id):
        img = self.img_aug(obj_id,bg_id)
        txt = self.txt_aug(obj_id,bg_id)
        return img,txt 
    
    def select_id(self, image_id, obj_bg):
        if obj_bg == "obj":
            obj_id = image_id
            bg_id  = random.choice(self.obj_bg_dict["bg"])
        elif obj_bg == "bg":
            bg_id  = image_id 
            obj_id = random.choice(self.obj_bg_dict["obj"])
        else:
            bg_id  = random.choice(self.obj_bg_dict["bg"])
            obj_id = random.choice(self.obj_bg_dict["obj"])

        return obj_id, bg_id
    
    def __call__(self, ann):  
        image_id = ann["image_id"].split("_")[-1]
        self.image_id = image_id

        obj_id, bg_id = self.select_id(image_id, self.image_info_dict[image_id]["obj_bg"])
        img, caption = self.mix(obj_id, bg_id)

        return img, caption
            
            

class RoMixGen_Img:
    def __init__(self, image_root, transform_after_mix, method = False,  img_mix_ratio=0.5):
    
        # Image 
        self.image_root          = image_root                   # preprocessed image for augmentation root 
        self.transform_after_mix = transform_after_mix          # transforms functions after augmentation 
        self.img_mix_ratio       = img_mix_ratio             # if True, mix image with image

    def __get_dict__(self, img_info):
        self.img_info_dict = img_info  # how large obj image resized   

    def bbox_point(self,bboxes):
        y_up = int(bboxes[1])
        y_down = y_up + int(bboxes[3])
        x_left = int(bboxes[0])
        x_right = x_left + int(bboxes[2])
        return x_left,x_right,y_up,y_down
    
    def __cut_obj__(self, bboxes, obj_img, obj_bg:str):
        x_left, x_right, y_up, y_down = self.bbox_point(bboxes)
        if obj_bg == "obj":
            result = np.array(obj_img)[y_up:y_down,x_left:x_right,:]
        elif obj_bg == "bg":
            result = np.array(obj_img)
            result[y_up:y_down,x_left:x_right,:] = 0
        return result 
    
    def __paste_obj__(self, bboxes, bg_img, obj_img):
        x_left, x_right, y_up, y_down = self.bbox_point(bboxes)
        bg_img[y_up:y_down, x_left:x_right, :] = obj_img
        return bg_img
    
    def __resize__(self, obj_img, bg_img, size=(384,384)):
        # img resize
        width_ratio, height_ratio = size[0] / obj_img.shape[1], size[1] / obj_img.shape[0]
        obj_img = cv2.resize(obj_img, size, interpolation=cv2.INTER_CUBIC)
        bg_img = cv2.resize(bg_img, size, interpolation=cv2.INTER_CUBIC)
        resized_bbox = [self.obj_inform["max_obj_bbox"][0] * width_ratio, self.obj_inform["max_obj_bbox"][1] * height_ratio, self.obj_inform["max_obj_bbox"][2] * width_ratio, self.obj_inform["max_obj_bbox"][3] * height_ratio]
        return obj_img, bg_img, resized_bbox
    
    def cutmixup(self,obj_inform,bg_inform):
        # Image load and convert to RGB 
        obj_img = np.array(Image.open(os.path.join(self.image_root, obj_inform['file_name'])).convert('RGB'))
        bg_img  = np.array(Image.open(os.path.join(self.image_root, bg_inform['file_name'])).convert('RGB'))
        
        # Preprocess for synthesize new image 
        bg_img =  self.__cut_obj__(self.bg_inform["max_obj_bbox"], bg_img, "bg") # Cutout object image part from background image 
        obj_img, bg_img, obj_bbox = self.__resize__(obj_img, bg_img,(512,512))   # resize image 
        obj_img_obj = self.__cut_obj__(obj_bbox, obj_img, "obj")                 
        obj_img_bg = self.__cut_obj__(obj_bbox, obj_img, "bg")
        img = (1 - self.img_mix_ratio) * obj_img_bg + self.img_mix_ratio * bg_img
        #regularization
        img = self.__paste_obj__(obj_bbox, img, obj_img_obj)
        # numpy array to PIL image
        img = Image.fromarray(img.astype(np.uint8))
        img = self.transform_after_mix(img)
        return img 
        
    def __call__(self,obj_info,bg_info):
        img = self.cutmixup(obj_info,bg_info)
        return img
    
class RoMixGen_Txt:
    def __init__(self,method = 'concat',txt_aug=False): 
        self.method = method
        self.pertur_list = get_method_chunk_train()
        self.txt_aug = txt_aug 
    
    def __get_dict__(self, img_info):
        self.img_info_dict = img_info
    
    def replace_word(self,captions,bg_cats,obj_cats):
        caption = np.random.choice(captions,1)[0]
        try:
            (bg_cat_id, bg_cat) = list(filter(lambda x : x[1] in caption.lower(), enumerate(bg_cats)))[0]
            caption = caption.lower().replace(bg_cat,obj_cats[bg_cat_id]).capitalize()
        except IndexError:
            caption = random.choice(obj_cats) + " " + caption
        return caption 
    
    def concat(self,obj_caption,bg_caption):
        text = np.random.choice([a + ' ' + b for a,b in zip(bg_caption,obj_caption)])
        return text 
    
    def conjunction_concat(self,obj_caption,bg_caption):
        conjunction_list = ['and', 'also', 'as well as', 'moreover', 'furthermore', 'in addition', 'besides', 'similarly', 'likewise', 'consequently', 'therfore', 'thus', 'hence']
        conjunction = np.random.choice(conjunction_list)
        text = np.random.choice([a + ' ' +conjunction+' '+ b for a,b in zip(bg_caption,obj_caption)])
        return text 
    
    def mix_concat(self,obj_caption,bg_caption):
        obj_cap, bg_cap = np.random.choice(obj_caption), np.random.choice(bg_caption)
        obj_cap_split = [obj_cap.split(" ")[i:i+3] for i in range(0, len(obj_cap.split(" ")), 3)]
        bg_cap_split = [bg_cap.split(" ")[i:i+3] for i in range(0, len(bg_cap.split(" ")), 3)]

        result = [x for pair in zip(obj_cap_split, bg_cap_split) for x in pair] + obj_cap_split[len(bg_cap_split):] + bg_cap_split[len(obj_cap_split):]
        result = " ".join(item for sublist in result for item in sublist)
        return result
    
    def txt_shuffle(self,txt1,txt2):
        txt1 = txt1.split(' ')
        txt2 = txt2.split(' ')
        new_txt = list(chain(*zip(txt1,txt2)))
        
        if len(txt1) > len(txt2):
            new_txt = " ".join(new_txt + txt1[int(len(new_txt)/2):])
        
        elif len(txt2) > len(txt1):
            new_txt = " ".join(new_txt + txt2[int(len(new_txt)/2):])
        else:
            new_txt = " ".join(new_txt)

        return new_txt
    
    def __call__(self,obj_id,bg_id):
        # object 와 background의 최대 카테고리 정보 
        obj_cat = self.img_info_dict[obj_id]["max_obj_cat"] + self.img_info_dict[obj_id]["max_obj_super_cat"]
        bg_cat = self.img_info_dict[bg_id]["max_obj_cat"] + self.img_info_dict[bg_id]["max_obj_super_cat"]
        
        # object와 background의 caption 
        bg_caption = self.img_info_dict[bg_id]["captions"]
        obj_caption = self.img_info_dict[obj_id]['captions'] 
        
        
        if self.method == 'replace':
            new_caption = self.replace_word(bg_caption,bg_cat,obj_cat)
            
        elif self.method == 'concat':
            new_caption = self.concat(obj_caption,bg_caption)
            
        elif self.method == 'conjconcat':
            new_caption = self.conjunction_concat(obj_caption,bg_caption)
            
        elif self.method == 'txtshuffle':   
            new_caption = np.random.choice([self.txt_shuffle(bg,obj) for bg,obj in zip(bg_caption,obj_caption)])
            
        elif self.method == 'mixconcat':
            new_caption = self.mix_concat(obj_caption,bg_caption)       
        
        if self.txt_aug:
            pertur = np.random.choice(self.pertur_list)
            new_caption = pertur(new_caption)
        
        return new_caption