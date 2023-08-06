import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models import load_model_vqa 

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from arguments import parser 

from scheduler import create_scheduler
from optim import create_optimizer
import wandb 
from augmentation import mixgen 

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(config, delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # Mixgen 
        if config['mixgen']:
            num = int(data_loader.batch_size/2)
            images,text = mixgen(image,list(question),num)
        
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      
        
        # Tokenizing 
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        # Model Inference 
        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)        
        
        # Optimizing 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        # Logging 
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size) 
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(config, delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])      
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            _, pred = topk_prob.max(dim=0)
            result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[topk_id[pred]]})   

    return result

def eval_text(args, config):
    from perturbation.text_perturbation import get_method_chunk
    from dataset.vqa_dataset import vqa_pertur_dataset
    
    pertur_list = get_method_chunk()
        
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())
        
    model, model_without_ddp, tokenizer = load_model_vqa(args, config, device)
    
    #### Dataset #### 
    print("Creating retrieval dataset") 
    for pertur in pertur_list:
        config['pertur'] = str(pertur).split(' ')[1]
        print(pertur)
        train_dataset, _ = create_dataset(
                                        dataset = 'vqa',
                                        config  = config
                                        )    
        
        test_dataset = vqa_pertur_dataset(
                                        ann_file = config['train_file'], 
                                        img_size = config['image_res'], 
                                        vqa_root = config['vqa_root'],
                                        vg_root  = config['vg_root'],
                                        txt_pertur = pertur, 
                                        answer_list = config['answer_list']
                                        ) 

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler([train_dataset, test_dataset], [True, False], num_tasks, global_rank) 
        else:
            samplers = [None, None, None]
        
        _, _, test_loader = create_loader([train_dataset, test_dataset],samplers,
                                            batch_size=[config['batch_size_train'],config['batch_size_test']],
                                            num_workers=[0,0],
                                            is_trains=[True, False], 
                                            collate_fns=[vqa_collate_fn,None]) 
        #### Wandb init #### 
        if utils.is_main_process(): 
            if config['wandb']['wandb_use']:
                wandb.init(project="RobustMixGen",name=config['exp_name']+'-'+str(pertur).split(' ')[1],config=config)
                
        #### Evaluation #### 
        print("Start Evaluation")
        start_time = time.time()  
        for epoch in range(0,1):  
            vqa_result = evaluation(model, test_loader, tokenizer, device, config)        
            result_file = save_result(
                result     = vqa_result,
                result_dir = args.result_dir,
                filename   = f"vqa_result_Image_{str(pertur).split(' ')[1]}"
                )
            
            
            dist.barrier()     
            torch.cuda.empty_cache()
    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str)) 
        wandb.finish()

def eval_image(args, config):
    from perturbation.image_perturbation import get_method_chunk
    from dataset.vqa_dataset import vqa_pertur_dataset
    
    pertur_list = get_method_chunk()
        
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())
        
    model, model_without_ddp, tokenizer = load_model_vqa(args, config, device)
    
    #### Dataset #### 
    print("Creating retrieval dataset") 
    for pertur in pertur_list:
        config['pertur'] = str(pertur).split(' ')[1]
        print(pertur)
        train_dataset, _ = create_dataset('vqa', config)  
        test_dataset = vqa_pertur_dataset(config['train_file'], config['image_res'], config['vqa_root'], config['vg_root'],
                                          img_pertur=pertur, answer_list=config['answer_list']) 

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler([train_dataset, test_dataset], [True, False], num_tasks, global_rank) 
        else:
            samplers = [None, None, None]
        
        _, test_loader = create_loader([train_dataset, test_dataset], samplers,
                                            batch_size=[config['batch_size_train'], config['batch_size_test']],
                                            num_workers=[0,0],
                                            is_trains=[True, False], 
                                            collate_fns=[vqa_collate_fn,None]) 
        #### Wandb init #### 
        if utils.is_main_process(): 
            if config['wandb']['wandb_use']:
                wandb.init(project="RobustMixGen",name=config['exp_name']+'-'+str(pertur).split(' ')[1],config=config)
                
        #### Evaluation #### 
        print("Start Evaluation")
        start_time = time.time()  
        for epoch in range(0,1):  
            vqa_result = evaluation(model, test_loader, tokenizer, device, config)        
            result_file = save_result(
                result     = vqa_result,
                result_dir = args.result_dir,
                filename   = f"vqa_result_Image_{str(pertur).split(' ')[1]}"
                )
            
            
            dist.barrier()     
            torch.cuda.empty_cache()
    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str)) 
        wandb.finish()

def main(args, config):
        
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    utils.set_seed(seed)
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
        
    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset(
        dataset = 'vqa',
        config  = config
        )   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(
                        datasets    = datasets, 
                        shuffles    = [True, False],
                        num_tasks   = num_tasks,
                        global_rank = global_rank
                        )         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(
        datasets    = datasets,
        samplers    = samplers,
        batch_size  = [config['batch_size_train'],config['batch_size_test']],
        num_workers = [0,0],
        is_trains   = [True, False], 
        collate_fns = [vqa_collate_fn,None]
        ) 

    
    
    #### Wandb #### 
    if utils.is_main_process(): 
        if config['wandb']['wandb_use']:
            wandb.init(project="RobustMixGen",name=config['exp_name'],config=config)

    #### Model #### 
    print("Creating model")
    model, model_without_ddp, tokenizer = load_model_vqa(
                                                        args   = args, 
                                                        config = config, 
                                                        device = device
                                                        )  
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(
                                args  = arg_opt, 
                                model = model
                                )
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(
                                        args      = arg_sche, 
                                        optimizer = optimizer
                                        )          
    
    #### Start Training #### 
    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
        
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_stats = train(
            model        = model,
            data_loader  = train_loader,
            optimizer    = optimizer,
            tokenizer    = tokenizer,
            epoch        = epoch,
            warmup_steps = warmup_steps,
            device       = device,
            scheduler    = lr_scheduler,
            config       = config
            )  

        if utils.is_main_process():               
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "main_log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                
            if config['wandb']['wandb_use']:
                wandb.log(log_stats)    
                
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  

        dist.barrier()   
  
    vqa_result = evaluation(
                            model       = model, 
                            data_loader = test_loader, 
                            tokenizer   = tokenizer, 
                            device      = device,
                            config      = config
                            )
    
    result_file = save_result(
                            result     = vqa_result,
                            result_dir = args.result_dir,
                            filename  = 'vqa_result_epoch%d'%epoch
                            )
                     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    config = parser()
    

    config.args.result_dir = os.path.join(config.args.output_dir, 'result')

    Path(config.args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.args.result_dir).mkdir(parents=True, exist_ok=True)
    
    args = config.args 
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)