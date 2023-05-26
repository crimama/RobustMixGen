import json

from PIL import Image
from torchvision import transforms

from .romixgen import MiX, RoMixGen_Img, RoMixGen_Txt
from dataset.randaugment import RandomAugment


def create_romixgen(config):
    if config['romixgen']['image']['hard_aug']:
        transform_after_mix     = transforms.Compose([                        
                                            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                                            transforms.RandomHorizontalFlip(),
                                            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                                            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                            ])  
    else:
        transform_after_mix = transforms.Compose([
                                            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                        ])
    
    
    #image_info_dict        = json.load(open(config['img_info_json']))

    
    img_func = RoMixGen_Img(image_root            = config['image_root'],
                            transform_after_mix   = transform_after_mix,
                            method                = config['romixgen']['image']['method'],
                            obj_bg_mix_ratio      = config['romixgen']['base']['obj_bg_mix_ratio'],) # obj image, bg image mix ratio (lambda value) 


    try:
        txt_func = RoMixGen_Txt(method = config['romixgen']['text']['method'],
                                txt_aug = config['romixgen']['text']['txt_aug'])
    except KeyError:
        
        txt_func = RoMixGen_Txt(method = config['romixgen']['text']['method'],
                                txt_aug = False)
         
    romixgen = MiX( image_info          = config['romixgen']['base']['img_info_json'],
                    img_aug_function    = img_func,
                    txt_aug_function    = txt_func,
                    obj_bg_threshold    = config['romixgen']['base']['obj_bg_threshold'],
                    bg_center_threshold = config['romixgen']['base']['bg_center_threshold'],
                    normal_image_root   = config['image_root'],
                    normal_transform    = transform_after_mix)
    
    return romixgen