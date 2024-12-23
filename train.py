"""
add_date: 23/6/29
author: LHY
构建benchmark

"""


from utils.scheduler import Scheduler
from utils.show_log import show_loss
from utils.utils import *

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as tdata 

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup


import os, sys, time, argparse, re, random
import os.path as osp
import numpy as np
from pathlib import Path

import data
try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
except:
    print('no torch_geometric')

# from test import sn6loc_test
from test import retrieval_test

from models import *


def train(model, dataloader, criterions, optimizer, scheduler, scaler,
          recorder, evaluater, start_epoch, exp_name, device, args):

    print('==> Start Training...')
    for epoch in range(start_epoch, args.train.epochs):

        print(f'===== Experiment: {exp_name} ====')

        ###DEBUG####
        # if epoch == 0:
        #     print("--------------DEBUG--------------")
        #     evaluater.evaluation(model)
        #     print("-------NO Bug in testing!--------")


        # initial everything
        model.train()
        optimizer.zero_grad(set_to_none=True) # Zero gradients for first step
        recorder.reset() #init record
        current_lr = scheduler.get_last_lr()

        
        for batch_idx, data in enumerate(dataloader):
            start_t = time.time()

            if scaler:
                with autocast():
                    data = move_to(data, device)
                    data = model(data)
                    if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1: 
                        loss = sum([cri(data, model.module.logit_scale.exp(), args) for cri in criterions])
                    else:
                        loss = sum([cri(data, model.logit_scale.exp(), args) for cri in criterions])
                
                scaler.scale(loss).backward()

                # Gradient clipping 
                if args.train.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.train.clip_grad)  
                
                # Update model parameters (weights)
                scaler.step(optimizer) 
                scaler.update()
                # Zero gradients for next step
                optimizer.zero_grad()      

                # Scheduler
                if args.train.scheduler == "polynomial" or args.train.scheduler == "cosine" or args.train.scheduler ==  "constant":
                    scheduler.step()  

            else:
                data = move_to(data, device)
                data = model(data)
                if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1: 
                    loss_A = criterions(data, model.module.logit_scale, args) 
                else:
                    args.train.this_epoch = epoch
                    loss = criterions(data, model.logit_scale, args) 
                
                loss.backward(retain_graph=False)

                # Gradient clipping 
                if args.train.clip_grad:
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.train.clip_grad)   

                # Update model parameters (weights)
                optimizer.step() 
                # Zero gradients for next step
                optimizer.zero_grad()

                # Scheduler
                if args.train.scheduler == "polynomial" or args.train.scheduler == "cosine" or args.train.scheduler ==  "constant":
                    scheduler.step()


            # record model parameter& grad
            params_list = list(zip(*model.named_parameters()))[-1]
            grads = torch.cat([x.grad.flatten() if x.grad is not None else x.new_zeros(1) for x in params_list]) # model grad
            params =  torch.cat([x.data.flatten() if x.grad is not None else x.new_zeros(1) for x in params_list]) # model parameters

            # check the update of network 
            if torch.isnan(params).sum() >0 or torch.isnan(grads).sum() >0:
                print(params)
                print(grads)

            # Record metric
            recorder.update('Loss', loss.item(), args.train.batch_size)
            recorder.update('FPS', args.train.batch_size/(time.time() - start_t))

            if batch_idx % 10 == 0: #show record
                rcd_list = [f'Epoch[{epoch}][{batch_idx}/{len(trainloader)}]',
                            f'lr:{current_lr[0]:.6f}']
                rcd_list.extend(recorder.display())
                print(' '.join([rcd for rcd in rcd_list])) 
                recorder.reset()
                
            #for visualization
            if vis_path is not None and batch_idx % 100 == 0 and epoch % 5 == 0:
                recorder.param_vis(batch_idx)

        # save model parameters
        if ((epoch+1) % args.train.save_epoch == 0) or (epoch+1 == args.train.epochs):
            state = {
                'model': model.state_dict(),
                'epoch': epoch+1,
            }
            torch.save(state, osp.join(checkpoint_path, f'{epoch+1}_param.t'))
            print('params saved on:' +  osp.join(checkpoint_path, f'{epoch+1}_param.t'))

        # evaluation
        if ((epoch+1) % args.train.eval_epoch == 0) or (epoch+1 == args.train.epochs): 
            evaluater.evaluation(model)

    print(f'======= Experiment: {exp_name} is finished! ======')

def parse():
    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
    parser.add_argument('cfg', default='configs/split.yml', type=str, help='yaml config path')
    parser.add_argument('--mixed_precision',default=True, type=bool)
    parser.add_argument('--gpus', '-g', default='0,1,2,3', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--resume', type=str,
                        help='load model to resume from checkpoint')
    parser.add_argument('--result_path', default='outputs', type=str,
                        help='model save path')
    parser.add_argument('--extend', action='store_true', 
                        help='use the extend 160k satellite gallery')
    args = parser.parse_args()
    opt = update_args(args)
    return opt


if __name__ == '__main__':
    home = os.path.expanduser('~')
    opt = parse()
    #set train environments
    

    # torch.cuda.set_device(opt.gpu)

    # print('gpu:',os.environ["CUDA_VISIBLE_DEVICES"])
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)   # 为所有GPU设置随机种子
    np.random.seed(1)
    random.seed(1)
    device = f'cuda' if torch.cuda.is_available() else 'cpu' 
    cudnn.benchmark = True #acclerate convolution

    #set paths
    exp_name = opt.cfg.split('/')[-1].split('.')[0]
    opt.train.data_path = osp.join(home, opt.train.data_path) #get absolute path
    opt.eval.data_path = osp.join(home, opt.eval.data_path)
    log_path = osp.join(opt.result_path, exp_name)
    vis_path = None
    checkpoint_path = log_path

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True) 

    #output path & print options
    sys.stdout = Logger(osp.join(log_path, f'Train_{opt.train.dataset}_{time.asctime()}.log'))
    print(f"==========\nArgs:{opt}\n==========")

    # dataloader
    print('==> Loading data..')
    trainset = DATASET[opt.train.dataset](**opt.train)
    trainloader = tdata.DataLoader(trainset, batch_size=opt.train.batch_size,
                                shuffle=True, num_workers=opt.workers, drop_last=True)
    
    if opt.train.custom_sampling:
        trainloader = tdata.DataLoader(trainset, batch_size=opt.train.batch_size,
                                shuffle=False, num_workers=opt.workers, drop_last=True, pin_memory=True)
        trainloader.dataset.shuffle()
    else:
        trainloader = tdata.DataLoader(trainset, batch_size=opt.train.batch_size,
                                shuffle=True, num_workers=opt.workers, drop_last=True, pin_memory=True)

    sat_mode = 'sat_160k' if opt.extend else 'sat'
    dro_mode = 'dro_split' if opt.train.split < 1 else 'dro'
    queryset = DATASET[opt.eval.dataset](mode=dro_mode, **opt.eval)
    gallset = DATASET[opt.eval.dataset](mode=sat_mode, **opt.eval)

    query_loader = tdata.DataLoader(queryset, batch_size=opt.eval.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)
    gall_loader = tdata.DataLoader(gallset, batch_size=opt.eval.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)

    #test loader
    evaluater = retrieval_test(opt, query_loader, gall_loader, device)

    #build model
    start_epoch = 0
    print('Building model..')
    model = MODEL[opt.model.name](**opt.model)
    # load the previous model
    if opt.resume:
        load_param = opt.resume
        print('Resuming from checkpoint..')
        if os.path.isfile(load_param):
            print(f'Loading checkpoint from {load_param}')
            checkpoint = torch.load(load_param)
            if 'epoch' in checkpoint:  
                start_epoch = checkpoint['epoch']
                print(f'Loaded checkpoint in epoch {checkpoint["epoch"]}')
            if 'model' in checkpoint:
                param = checkpoint['model']
                model.load_state_dict(param)
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f'Warning: No checkpoint found at {load_param}')

    #-----------------------------------------------------------------------------#
    # Optimizer                                                                   #
    #-----------------------------------------------------------------------------#
    optimizer = model.build_opt(opt) #optimizer for main model



    # Data parallel
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    opt.gpu_ids = tuple(map(int, opt.gpus.split(',')))
    print(f"All GPUs: {torch.cuda.device_count()},  select GPUs: {opt.gpu_ids}")  
    if torch.cuda.device_count() > 1 and len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model.to(device)

    #record
    recorder = Recorder({"FPS":'d', "Loss":'f'})

    #loss function
    criterions = LOSS[opt.train.loss[0]](opt.train.loss_w[0], opt, recorder, device) 
    print("Criterion:", ", ".join([type(cri).__name__ for cri in [criterions,]]))
    print(f'Device: {device}')

    if opt.train.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None

    
    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(trainloader) * opt.train.epochs
    warmup_steps = len(trainloader) * opt.train.warm_epochs
       
    if opt.train.scheduler == "polynomial":
        print(f"\nScheduler: polynomial - max LR: {opt.train.lr} - end LR: {opt.train.lr_end}")  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = opt.train.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif opt.train.scheduler == "cosine":
        print(f"\nScheduler: cosine - max LR: {opt.train.lr}")   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif opt.train.scheduler == "constant":
        print(f"\nScheduler: constant - max LR: {opt.train.lr}")   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        # scheduler = Scheduler(optimizers, step_size=opt.train.lr_step_size, gamma=opt.train.lr_decay)
        
    print(f"Warmup Epochs: {opt.train.warm_epochs} - Warmup Steps: {warmup_steps}")
    print(f"Train Epochs:  {opt.train.epochs} - Train Steps:  {train_steps}")

    
    train(model, trainloader, criterions, optimizer, scheduler, scaler, 
          recorder, evaluater, start_epoch, exp_name, device, opt)
    # show_loss(log_path, param='top-1', file_type='log', keyword='Retrieval', vis=False)