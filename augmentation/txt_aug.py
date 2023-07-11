import os 
from itertools import chain    
import random 
import numpy as np 
from perturbation.text_perturbation import  get_method_chunk_train

class txt_aug_function:
    def __init__(self, txt_method:str='conjunction_concat', txt_pertur:bool=False):
        self.txt_method = get_txt_method(txt_method)
        self.txt_pertur = txt_pertur 
        self.txt_pertur_list = get_method_chunk_train()
        
    def __call__(self, obj_info, bg_info):
        caption = self.txt_method(obj_info, bg_info)
        if self.txt_pertur:
            pertur = np.random.choice(self.txt_pertur_list)
            caption = pertur(caption)
        return caption 

def get_txt_method(txt_method:str):
    if txt_method == 'concat':
        return concat    
    elif txt_method == 'conjunction_concat':
        return conjunction_concat
    elif txt_method == 'mix_concat':
        return mix_concat
    elif txt_method == 'txt_shuffle':
        return txt_shuffle
    else:
        raise KeyError
    
def replace_word(captions,bg_cats,obj_cats):
    caption = np.random.choice(captions,1)[0]
    try:
        (bg_cat_id, bg_cat) = list(filter(lambda x : x[1] in caption.lower(), enumerate(bg_cats)))[0]
        caption = caption.lower().replace(bg_cat,obj_cats[bg_cat_id]).capitalize()
    except IndexError:
        caption = random.choice(obj_cats) + " " + caption
    return caption 

def concat(obj_info,bg_info):    
    text = np.random.choice([a + ' ' + b for a,b in zip(bg_info['captions'],obj_info['captions'])])
    return text 

def conjunction_concat(obj_info,bg_info):
    conjunction_list = ['and', 'also', 'as well as', 'moreover', 'furthermore', 'in addition', 'besides', 'similarly', 'likewise', 'consequently', 'therfore', 'thus', 'hence']
    conjunction = np.random.choice(conjunction_list)
    text = np.random.choice([a.split('.')[0] + ' ' +conjunction+' '+ b for a,b in zip(bg_info['captions'],obj_info['captions'])])
    return text 

def mix_concat(obj_info,bg_info):
    obj_cap, bg_cap = np.random.choice(obj_info['captions']), np.random.choice(bg_info['captions'])
    obj_cap_split = [obj_cap.split(" ")[i:i+3] for i in range(0, len(obj_cap.split(" ")), 3)]
    bg_cap_split = [bg_cap.split(" ")[i:i+3] for i in range(0, len(bg_cap.split(" ")), 3)]

    result = [x for pair in zip(obj_cap_split, bg_cap_split) for x in pair] + obj_cap_split[len(bg_cap_split):] + bg_cap_split[len(obj_cap_split):]
    result = " ".join(item for sublist in result for item in sublist)
    return result

def txt_shuffle(obj_info,bg_info):
    txt1 = txt1.split(' ')
    txt2 = txt2.split(' ')
    new_txt = list(chain(*zip(bg_info['captions'],obj_info['captions'])))
    
    if len(txt1) > len(txt2):
        new_txt = " ".join(new_txt + txt1[int(len(new_txt)/2):])
    
    elif len(txt2) > len(txt1):
        new_txt = " ".join(new_txt + txt2[int(len(new_txt)/2):])
    else:
        new_txt = " ".join(new_txt)

    return new_txt