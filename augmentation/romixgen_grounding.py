from .base_romixgen import BaseRomixgen


class GroundingRomixgen(BaseRomixgen):
    def __init__(self, image_info_dir: str, image_root:str, transform, image_mix_ratio:float,
                    txt_method:str, txt_pertur:bool, obj_bg_threshold:float):
        super(GroundingRomixgen, self).__init__(
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
        text  = self.txt_aug([obj_info['text']],[bg_info['text']])
        return image, text 
    
    def __call__(self, image_id:str): 
        obj_id, bg_id, image_id = self.select_id(image_id)
        image, text = self.mix(obj_id, bg_id)
        return image, text, image_id
        