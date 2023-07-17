import os 
import numpy as np 
import cv2 
from PIL import Image 

class img_aug_function:
    '''
    bbox : X,Y,H,W
    '''
    def __init__(self, image_root:str, transform, image_mix_ratio:float):
    
        # Image 
        self.image_root          = image_root                   # preprocessed image for augmentation root 
        self.transform_after_mix = transform          # transforms functions after augmentation 
        self.img_mix_ratio       = image_mix_ratio             # if True, mix image with image

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
    
    def __load_resize__(self, obj_info, bg_info, size=(384,384)):
        # load img 
        obj_img = np.array(Image.open(os.path.join(self.image_root, obj_info['file_name'])).convert('RGB'))
        bg_img  = np.array(Image.open(os.path.join(self.image_root, bg_info['file_name'])).convert('RGB'))
        # img resize
        width_ratio, height_ratio = size[0] / obj_img.shape[1], size[1] / obj_img.shape[0]
        obj_img = cv2.resize(obj_img, size, interpolation=cv2.INTER_CUBIC)
        bg_img = cv2.resize(bg_img, size, interpolation=cv2.INTER_CUBIC)
        resized_bbox = [obj_info["max_obj_bbox"][0] * width_ratio, obj_info["max_obj_bbox"][1] * height_ratio, obj_info["max_obj_bbox"][2] * width_ratio, obj_info["max_obj_bbox"][3] * height_ratio]
        return obj_img, bg_img, resized_bbox
    
    def cutmixup(self,obj_info,bg_info):
        '''
        info 가 필요한 항목 
            - file_name
            - max_obj_bbox
        '''
        # Preprocess for synthesize new image 
        obj_img, bg_img, obj_bbox = self.__load_resize__(obj_info,bg_info,(512,512))   # resize image 
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