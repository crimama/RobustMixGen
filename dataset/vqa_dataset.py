import os
import json
import random
import numpy as np 
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question, pertur_check, get_transform



class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, 
                romixgen: object, romixgen_true: bool = True ,romixgen_prob: float = 0.1,
                eos='[SEP]', split="train", max_ques_words=30, answer_list=''):
        
        self.split = split        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        
        self.romixgen = romixgen
        self.romixgen_true = romixgen_true
        self.romixgen_prob = romixgen_prob 
        
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
            self.answer_list = json.load(open(answer_list,'r'))    
                
        
    def __len__(self):
        return len(self.ann)
    
    def __load_image__(self,ann):
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image']) 
        image = Image.open(image_path).convert('RGB')  
        image = self.transform(image) 
        return image 
    
    def __test_load__(self, ann):
        image = self.__load_image__(ann)
        question = pre_question(ann['question'],self.max_ques_words)   
        question_id = ann['question_id']
        return image, question, question_id 
    
    def __train_load__(self, ann):
        image = self.__load_image__(ann)
        question = pre_question(ann['question'],self.max_ques_words)   
        if ann['dataset']=='vqa':                
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

        elif ann['dataset']=='vg':
            answers = [ann['answer']]
            weights = [0.5]  
        answers = [answer+self.eos for answer in answers]
            
        return image, question, answers, weights
    
    def __getitem__(self, index):    
        ann = self.ann[index]

        if self.split == 'test':            
            image, question, question_id = self.__test_load__(ann)            
            return image, question, question_id 
        
        elif self.split == 'train':
            if (self.romixgen_true) & (random.random() < self.romixgen_prob):
                image, question, answers, weights = self.romixgen(ann)
            else:
                image, question, answers, weights = self.__train_load__(ann)    
            return image, question, answers, weights 


class vqa_pertur_dataset(vqa_dataset):
    def __init__(self, ann_file, img_size, vqa_root, vg_root, eos='[SEP]', split="test",
                img_pertur=None, txt_pertur=None, max_ques_words=30, answer_list=''):
        
        self.split = split        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        self.transform      = get_transform(img_size)
        self.vqa_root       = vqa_root
        self.vg_root        = vg_root
        self.max_ques_words = max_ques_words
        self.eos            = eos
        
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
            self.answer_list = json.load(open(answer_list,'r'))
                
        self.img_pertur = pertur_check(img_pertur)
        self.txt_pertur = pertur_check(txt_pertur)
            
    def __getitem__(self, index):
        
        ann = self.ann[index]
        
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])
            
        image = Image.open(image_path).convert('RGB')   
        image = self.img_pertur(np.array(image),severity=2).astype(np.uint8)
        image = self.transform(Image.fromarray(image).convert('RGB'))     
        
        question = pre_question(ann['question'],self.max_ques_words)   
        question = self.txt_pertur(question)
        question_id = ann['question_id']            
        return image, question, question_id
    

def cal_answer_weights(answers:list, dataset:str, eos:str) -> list:
    if dataset=='vqa':                
        answer_weight = {}
        for answer in answers:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1/len(answers)
            else:
                answer_weight[answer] = 1/len(answers)

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

    elif dataset=='vg':
        answers = [answers]
        weights = [0.5]  
    answers = [answer+eos for answer in answers]
        
    return answers, weights