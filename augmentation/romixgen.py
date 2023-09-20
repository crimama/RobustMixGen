import torch 
import numpy as np 

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

class Romixgen:
    def __init__(self, device, 
                 transform,
                 score_thresh:float = 0.8, alpha:float = 0.5, 
                 batch_size:int = 64, mix_ratio:float = 0.25,
                 ):
        self.detector = self.get_detector(score_thresh, device)
        self.alpha = alpha 
        self.num_mix = int(batch_size * mix_ratio)
        self.batch_size = batch_size 
        self.transform = transform 
        
    def get_detector(self, score_thresh:float, device:str):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = device 
        self.cfg = cfg 
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        predictor.model.to(device)
        return predictor 
    
    def preprocess(self, x):
        height, width = x.shape[:2]
        image = self.detector.aug.get_transform(x).apply_image(x) # resize 
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return inputs 
    
    def postprocess(self, prediction):
        boxes  = prediction._fields['pred_boxes'].tensor
        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        # in case no object in pic
        if len(areas) == 0:
            return { 'bbox' : None, 'label' : None }
        # choose one object among objects 
        # target_idx = torch.max(areas)
        # target_idx = torch.where(areas ==torch.median(areas))[0].item()
        target_idx = torch.argmin(areas)
        box = boxes[target_idx].type(torch.int).numpy()

        classes = prediction._fields['pred_classes']
        label_list = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes",None)
        classes = prediction.pred_classes.tolist()
        label = label_list[classes[target_idx]]
        output = { 'bbox' : box, 'label' : label }
        return output 
    
    def img_mix(self, output, img_obj, img_bg):
        bbox = output['bbox']
        if bbox is None:
            return img_obj, None 
        x0,y0, x1,y1 = bbox 
        

        obj = img_obj[y0:y1, x0:x1]    
        mix_img = img_obj *0.5 + img_bg * 0.5
        mix_img[y0:y1, x0:x1] = img_bg[y0:y1, x0:x1] * 0.25 + obj  * 0.75
        label = output['label']
        return mix_img, label 
    
    def txt_mix(self, obj_label, bg_label, word):
        if word is None:
            return bg_label 
        else:
            mix_label = f"{obj_label} focusing on {word} and {bg_label}"
            return mix_label 

    @torch.no_grad()
    def __call__(self, imgs, labels):
        img_objs = imgs[:self.num_mix]
        img_bgs = imgs[int(self.batch_size/2):int(self.batch_size/2)+self.num_mix]
        # preprocess 
        inputs = [self.preprocess(x) for x in imgs[:self.num_mix]]
        # Inference 
        outputs = self.detector.model(inputs)
        # Postprocess 
        outputs = [self.postprocess(x['instances'].to('cpu')) for x in outputs]
        outputs = np.array([self.img_mix(output, img_obj, img_bg) for output, img_obj, img_bg  in zip(outputs,img_objs,img_bgs)])
        imgs[:self.num_mix] = outputs[:,0]
        
        # Caption mix
        word_objects = outputs[:,1]
        obj_labels = labels[:self.num_mix]
        bg_labels = labels[int(self.batch_size/2):int(self.batch_size/2)+self.num_mix]
        labels[:self.num_mix] = [self.txt_mix(obj_l,bg_l,word) for obj_l, bg_l, word in zip(obj_labels, bg_labels, word_objects)]
        
        torch.cuda.empty_cache()
        return imgs, labels