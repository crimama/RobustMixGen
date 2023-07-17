from PIL import Image
from torchvision import transforms
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
    if config.TASK == 'VQA':
        romixgen = __import__('augmentation').__dict__[f"{config.TASK}Romixgen"](
                vg_info_dir          = config['romixgen']['base']['vg_info_dir'],
                vqa_info_dir         = config['romixgen']['base']['vqa_info_dir'],
                vqa_root             = config['vqa_root'],
                vg_root              = config['vg_root'],
                transform            = transform_after_mix, 
                image_mix_ratio      = config['romixgen']['base']['img_mix_ratio'],
                txt_method           = config['romixgen']['text']['method'],
                txt_pertur           = config['romixgen']['text']['txt_aug'],
                txt_eos              = config['eos'],
                vg_obj_bg_threshold  = config['romixgen']['base']['vg_obj_bg_threshold'],
                vqa_obj_bg_threshold = config['romixgen']['base']['vqa_obj_bg_threshold']
                )
    else:
        romixgen = __import__('augmentation').__dict__[f"{config.TASK}Romixgen"](
                    image_info_dir  = config['romixgen']['base']['img_info_json'],
                    image_root      = config['image_root'],
                    transform       = transform_after_mix, 
                    image_mix_ratio = config['romixgen']['base']['img_mix_ratio'],
                    txt_method      = config['romixgen']['text']['method'],
                    txt_pertur      = config['romixgen']['text']['txt_aug'],
                    obj_bg_threshold= config['romixgen']['base']['obj_bg_threshold']
                    )
        
    return romixgen

