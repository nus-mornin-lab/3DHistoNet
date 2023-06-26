import os, sys, time, warnings, glob, tqdm, h5py

warnings.filterwarnings('ignore')

import argparse
from argparse import ArgumentParser

from toolz import *
from toolz.curried import *
from itertools import islice

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist


from torch.utils.data import DataLoader

from techniques import BYOL, SIMCLR, SIMCLR3D
from models.ResNet import resnet50_baseline
from data.DataGenSSL2 import DataGen

from utils import GET_OPTIMIZER

def Parse():
    
    parser = ArgumentParser("pretrain", add_help = False)

    parser.add_argument("--technique", type=str, default="SIMCLR", help = "SIMCLR | BYOL")
    parser.add_argument("--dataPath", type=str, default="./data/datasets/processed/flatten")
    parser.add_argument("--ckptPath", type=str, default="./ckpt/pretrain/SIMCLR")
    parser.add_argument("--logPath", type=str, default="./log/pretrain")
            
    parser.add_argument("--gpuN", type=str, default="0")
    parser.add_argument("--cpuN", type=int, default=20) # per gpu
    parser.add_argument("--batchSize", type=int, default=256) # per gpu #256
    parser.add_argument("--epochN", type=int, default=251)
    parser.add_argument("--precision", type=str, default="half", help = "single | half")
    
    parser.add_argument("--optName", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    parser = first(parser.parse_known_args())
    
    # pick a gpu that has the largest space
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpuN
     
    return parser

def trainStep(data, ssl, optimizer, scaler):
    
    optimizer.zero_grad()
    
    if scaler is not None :
        
        with torch.cuda.amp.autocast():        
            
            # loss : Dict
            loss = ssl.forward(*data) 
            
            scaler.scale(loss["SIMCLR_LOSS"]).backward()
            scaler.step(optimizer)            
            scaler.update()
            
    else:
        
        loss = ssl.forward(*data) 
            
        loss["SIMCLR_LOSS"].backward()
        optimizer.step()
        
    return loss["SIMCLR_LOSS"]

@torch.no_grad()
def validStep(data, ssl):   
    
    loss = ssl.forward(*data)
    
    return loss["SIMCLR_LOSS"]

def train(trainDataLoader, validDataLoader, ssl, optimizer, config) :
    
    minLoss = 9999
    nIter   = 0     
    scaler  = torch.cuda.amp.GradScaler() if config.precision is "half" else None
    
    ssl.train()    
    for e in range(251, 251 + config.epochN):
        
        print("TRAINING...")
        ######################################################
        losses = []
        ssl.train()        
        print(f"epoch: {e}/{config.epochN}")
        for i, data in enumerate(tqdm.tqdm(trainDataLoader)):
                                 
            loss = trainStep(data, ssl, optimizer, scaler)
            losses.append(loss.item())
            nIter = nIter + 1
            
            if i % 30 == 0:
                lossStack = np.stack(losses)
                lossStack = float(str(np.mean(lossStack))[:4])
                
                print(f"(EPOCH {e}),  ({i}/{len(trainDataLoader)}) TRAIN LOSS  : {lossStack}")
            
        losses = np.stack(losses)
        loss = float(str(np.mean(losses))[:4])
        
        ######################################################        
        if config.technique == "BYOL":
            ssl.update_moving_average()
        
        if e % 10 == 0:
            torch.save(ssl.state_dict(), f"{config.ckptPath}/{e}.pt")
        
if __name__ == '__main__':
        
    # init configs
    ###################################################    
    config = Parse()
        
    # model, optmizer, ssl
    ###################################################
    net = resnet50_baseline(pretrained=None)
    opt = GET_OPTIMIZER(net.parameters(), config.optName, config.lr, 0)   
    
    ssl = {"BYOL"     : lambda : BYOL.BYOL,
           "SIMCLR"   : lambda : SIMCLR.SIMCLR,
           "SIMCLR3D" : lambda : SIMCLR3D.SIMCLR3D}[config.technique]()(net).cuda()    
    
    stateDict = torch.load(f"{config.ckptPath}/240.pt")
    stateDict = {k.replace("module.", "") : v for k,v in stateDict.items()}
    ssl.load_state_dict(stateDict)
    
    # dataloaders
    ################################################### 
    trainDataLoader = DataLoader(DataGen(config.dataPath,
                                         transform = ssl.genTask, 
                                         is2D      = "2D" in config.ckptPath,
                                         train     = True),
                                 shuffle     = True,
                                 batch_size  = config.batchSize,                                 
                                 pin_memory  = True,
                                 num_workers = config.cpuN)
    
    validDataLoader = DataLoader(DataGen(config.dataPath,
                                         transform = ssl.genTask,
                                         is2D      = "2D" in config.ckptPath,
                                         train     = False),
                                 shuffle     = False,
                                 batch_size  = config.batchSize,
                                 pin_memory  = True,
                                 num_workers = config.cpuN)
    
     # train & valid
    ###################################################
    train(trainDataLoader, validDataLoader, ssl, opt, config)
