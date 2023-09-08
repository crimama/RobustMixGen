from omegaconf import OmegaConf
import argparse
import os 
from easydict import EasyDict
def convert_type(value):
    # None
    if value == 'None':
        return None
    
    # list or tuple
    elif len(value.split(',')) > 1:
        return value.split(',')
    
    # bool
    check, value = str_to_bool(value)
    if check:
        return value
    
    # float
    check, value = str_to_float(value)
    if check:
        return value
    
    # int
    check, value = str_to_int(value)
    if check:
        return value
    
    return value

def str_to_bool(value):
    try:
        check = isinstance(eval(value), bool)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def str_to_float(value):
    try:
        check = isinstance(eval(value), float)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def str_to_int(value):
    try:
        check = isinstance(eval(value), int)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value

import ruamel.yaml as yaml
from easydict import EasyDict

def get_parser():
    parser = argparse.ArgumentParser(description='Roburst_Mixgen')
    parser.add_argument('--default_setting', type=str, default='configs/default.yaml', help='default config file')
    parser.add_argument('--task_setting', type=str, default=None, help='task config file')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args 

def parser(jupyter:bool = False, default_setting:str = None, task_setting:str = None):
    
    if jupyter:
        args = EasyDict({'default_setting': default_setting, 
                'task_setting'   : task_setting})
    else:
        args = get_parser()
        default_setting = args.default_setting 

    # load default config
    cfg = OmegaConf.load(args.default_setting)    
    
    if args.task_setting:
        cfg_task = OmegaConf.load(args.task_setting)
        cfg = OmegaConf.merge(cfg, cfg_task)
    
    # Update experiment name
    if cfg.TASK =='VQA':
        cfg.data_name = cfg.vqa_root.split('/')[2] + '&' + cfg.vg_root.split('/')[2]
        cfg.exp_name = cfg['TASK'] + '-' + cfg.data_name
    elif cfg.TASK == 'Pretrain':
        cfg.data_name = cfg.image_root
        cfg.exp_name = ''
    else:
        cfg.data_name = cfg.image_root.split('/')[2]
        cfg.exp_name = cfg['TASK'] + '-' + cfg.image_root.split('/')[2]
    
    # update cfg
    if not jupyter:
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            if k == 'exp_name':
                cfg.exp_name = f'{cfg.exp_name}-{v}'
            else:
                OmegaConf.update(cfg, k, convert_type(v), merge=True)  
    
    #! update exp_name in case romixgen 
    # if cfg['romixgen']['base']['romixgen_true']:
    #     cfg.exp_name = f'{cfg.exp_name}-romixgen'
    # else:
    #     pass 
    
    # if cfg['mixgen']:
    #     cfg.exp_name = f'{cfg.exp_name}-mixgen'
    # else:
    #     pass 
    
    # Output dir 
    if cfg.TASK in ['VQA', 'Pretrain']:
        cfg['args']['output_dir'] = os.path.join(cfg.args.output_dir, cfg.TASK)
    else:
        if cfg['romixgen']['base']['romixgen_true']:
            cfg['args']['output_dir'] = os.path.join(cfg.args.output_dir, cfg.TASK, cfg.image_root.split('/')[2], 'romixgen')
            
        elif cfg['mixgen']:
            cfg['args']['output_dir'] = os.path.join(cfg.args.output_dir, cfg.TASK, cfg.image_root.split('/')[2], 'mixgen')
            
        else:
            cfg['args']['output_dir'] = os.path.join(cfg.args.output_dir, cfg.TASK, cfg.image_root.split('/')[2], 'clean')
    
    cfg['args']['output_dir'] = os.path.join(cfg.args.output_dir,cfg.exp_name)
        
    cfg = EasyDict(OmegaConf.to_container(cfg))
    
    return cfg  
