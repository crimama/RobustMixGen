import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle
import wandb 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_ve import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import load_model_ve

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from arguments import parser 
from augmentation import mixgen 


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, wandb):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(config, delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    for i,(images, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # Mixgen 
        if config['mixgen']:
            num = int(data_loader.batch_size/2)
            images,text = mixgen(images,list(text),num)
    
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
        
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device) 
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss = model(images, text_inputs, targets=targets, train=True, alpha=alpha)    
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
            
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    

def eval_text(args, config):
    from perturbation.text_perturbation import get_method_chunk
    from dataset.ve_dataset import ve_pertur_dataset
    pertur_list = get_method_chunk()
    
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())
        
    model, model_without_ddp, tokenizer = load_model_ve(args, config, device)
    
    #### Dataset #### 
    print("Creating dataset") 
    for pertur in pertur_list:
        config['pertur'] = str(pertur).split(' ')[1]
        print(pertur)
        train_dataset, val_dataset, _ = create_dataset('ve', config)  
        test_dataset = ve_pertur_dataset(config['test_file'],config['image_res'],config['image_root'], txt_pertur=pertur)

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
        else:
            samplers = [None, None, None]
        
        _, _, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                            batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                            num_workers=[0,0,0],
                                            is_trains=[True, False, False], 
                                            collate_fns=[None,None,None])  
        
        #### Wandb init #### 
        if utils.is_main_process(): 
            if config['wandb']['wandb_use']:
                wandb.init(project="RobustMixGen",name=config['exp_name']+'-'+str(pertur).split(' ')[1],config=config)
                
        #### Evaluation #### 
        print("Start Evaluation")
        start_time = time.time()  
        for epoch in range(0,1):  
            test_stats = evaluate(model, test_loader, tokenizer, device, config)
            
            if utils.is_main_process():  
                    
                log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},                  
                            'epoch': epoch,
                            'pertur_type' : 'Text',
                            'pertur': str(pertur).split(' ')[1]
                            }
                with open(os.path.join(args.output_dir, "Eval_log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
                print(log_stats)
                if config['wandb']['wandb_use']:
                    wandb.log(log_stats)
                    
            
            dist.barrier()     
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))        
        wandb.finish()

def eval_image(args, config):
    from perturbation.image_perturbation import get_method_chunk
    from dataset.ve_dataset import ve_pertur_dataset
    pertur_list = get_method_chunk()
        
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())
    
    model, model_without_ddp, tokenizer = load_model_ve(args, config, device)
    
    #### Dataset #### 
    print("Creating dataset") 
    for pertur in pertur_list:
        config['pertur'] = str(pertur).split(' ')[1]
        print(pertur)
        train_dataset, val_dataset, _ = create_dataset('ve', config)  
        test_dataset = ve_pertur_dataset(config['test_file'],config['image_res'],config['image_root'], img_pertur = pertur)

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
        else:
            samplers = [None, None, None]
        
        _, _, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                            batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                            num_workers=[0,0,0],
                                            is_trains=[True, False, False], 
                                            collate_fns=[None,None,None])  
        
        #### Wandb init #### 
        if utils.is_main_process(): 
            if config['wandb']['wandb_use']:
                wandb.init(project="RobustMixGen",name=config['exp_name']+'-'+str(pertur).split(' ')[1],config=config)
                
        #### Evaluation #### 
        print("Start Evaluation")
        start_time = time.time()  
        for epoch in range(0,1):  
            test_stats = evaluate(model_without_ddp, test_loader, tokenizer, device, config)
            
            if utils.is_main_process():  
                    
                log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},                  
                            'epoch': epoch,
                            'pertur_type' : 'Image', 
                            'pertur': str(pertur).split(' ')[1]
                            }
                with open(os.path.join(args.output_dir, "Eval_log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
                print(log_stats)
                if config['wandb']['wandb_use']:
                    wandb.log(log_stats)
                    
            
            dist.barrier()     
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))        
        wandb.finish()

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(config, delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)   
        
        text = [list_unpack(x) for x in text]
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  

        prediction = model(images, text_inputs, targets=targets, train=False)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(float(accuracy.item()), n=images.size(0))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    #return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def list_unpack(x):
    if type(x) == list:
        return x[0]
    else:
        return x     
    
def main(args, config):
        
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    utils.set_seed(seed)

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('ve', config) 
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                            batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                            num_workers=[0,0,0],is_trains=[True,False,False], 
                                                            collate_fns=[None,None,None]
                                                            )
    #### wandb logging #### 
    if utils.is_main_process(): 
        if config['wandb']['wandb_use']:
            wandb.init(project="RobustMixGen",name=config['exp_name'],config=config)
            
    #### Model #### 
    model, model_without_ddp, tokenizer = load_model_ve(args, config, device)
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    
    print("Start training")
    start_time = time.time()
    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, wandb)  
            
        val_stats = evaluate(model, val_loader, tokenizer, device, config)
        test_stats = evaluate(model, test_loader, tokenizer, device, config)

        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        }

            with open(os.path.join(args.output_dir, "main_log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if config['wandb']['wandb_use']:
                wandb.log(log_stats)    

            if float(val_stats['acc'])>best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                best = float(val_stats['acc'])
                best_epoch = epoch
        
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "main_log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)         
            

if __name__ == '__main__':
    config = parser()
    args = config.args 
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    main(args, config)
    config['args']['checkpoint'] = os.path.join(config['args']['output_dir'],'checkpoint_best.pt')
