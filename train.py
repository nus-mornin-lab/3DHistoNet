from matplotlib import pyplot as plt

import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
from easydict import EasyDict

from toolz import *
from toolz.curried import *
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

from models.ResNet1D import ResNet1D
from models.Clam import CLAM

from data.DataGen import DataGen
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler as sampler

from utils import GET_OPTIMIZER, COMPUTE_METRIC, COMPUTE_LOSS, COMPUTE_WEIGHTS

def parse():

    parser = ArgumentParser()

    parser.add_argument("--dataPath", type=str, default="./data/datasets")
    parser.add_argument("--ckptPath", type=str, default="./ckpt/finetune")
    
    parser.add_argument("--featureN", type=int, default = 1024)
    parser.add_argument("--pretrain", type=str, default = "SIMCLR2D", help = "IMAGENET | SIMCLR | SIMCLRDCONV | SIMCLRF " )
    parser.add_argument("--mil", type=str, default = "mean", help = "mean | max | attention")
    parser.add_argument("--mode", type=str, default = "3D", help = "2D | 3D | 3D2 | 3D4")
    parser.add_argument("--diagnosis", type=str, default = "ER", help = "ER | PR | AR | HER2_IHC | KI67" )
    
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--weightType", type=str, default="sampler", help = "sampler | loss")
    parser.add_argument("--lr", type=float, default= 1e-4) # 1e-5
    parser.add_argument("--sliceNorm", type=bool, default= False) 
    
    parser.add_argument("--gpuN", type=str, default="2", help = "0|1|2|3") 
    parser.add_argument("--cpuN", type=int, default=6)
    parser.add_argument("--epochN", type=int, default=135)
    
    config = first(parser.parse_known_args())
    
    #if config.mode != "2D":
    #config.lr = 1e-5
    
    # pick a gpu that has the largest space
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"  
    os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuN
    
    return config

def trainStep(x, y, net, lossFn, optimizer):

    optimizer.zero_grad()
    
    logit, *_ = net.forward(x)
    
    loss = lossFn(logit, y)
    
    loss.backward()
    optimizer.step()
    
    return loss, logit

@torch.no_grad()
def validStep(x, y, net, lossFn):
    
    logit, *_ = net(x)
    
    loss = lossFn(logit, y)
    
    return loss, logit

def train(trainDataLoader, validDataLoader, net, lossFn, optimizer, config) :
    
    maxAuc = 0
    
    net.train()    
    for e in range(config.epochN):
        
        print("TRAINING...")
        ######################################################
        losses = []
        logits  = []
        targets = []        
        net.train()        
        print(f"epoch: {e}/{config.epochN}")
        for (x, y, *_) in trainDataLoader:
            
            loss, logit = trainStep(x.squeeze(0).cuda(),
                                    y.cuda(),
                                    net,
                                    lossFn,
                                    optimizer)
            
            losses.append(loss.item())
            targets.append(y.detach().cpu().numpy().squeeze(0))
            logits.append(logit.detach().cpu().numpy().squeeze(0))
                        
        losses  = np.stack(losses)
        logits  = np.stack(logits)
        targets = np.stack(targets)
        
        auc  = COMPUTE_METRIC(logits, targets)["aucs"]
        loss = float(str(np.mean(losses))[:4])
                
        print(f"TRAIN AUC  : {auc}")
        print(f"TRAIN LOSS : {loss}")
        ######################################################
        
        print("VALIDATING...")
        ######################################################
        losses  = []
        logits  = []
        targets = []
        net.eval()
        for (x, y, *_) in validDataLoader:
            
            loss, logit = validStep(x.squeeze(0).cuda(),
                                    y.cuda(),
                                    net,
                                    lossFn)
            
            losses.append(loss.item())
            targets.append(y.detach().cpu().numpy().squeeze(0))
            logits.append(logit.detach().cpu().numpy().squeeze(0))
        
        losses  = np.stack(losses)
        logits  = np.stack(logits)
        targets = np.stack(targets)
        
        auc  = COMPUTE_METRIC(logits, targets)["aucs"]
        loss = float(str(np.mean(losses))[:4])
        
        print(f"VALID AUC  : {auc}")
        print(f"VALID LOSS : {loss}")
    
        if maxAuc < auc :
            print(f"(saving) the current auc {auc} is bigger than the previous auc {maxAuc}")    
            directory = f"{config.ckptPath}/{config.pretrain}/{config.mode}/{config.mil}/{config.diagnosis}"
            os.makedirs(directory, exist_ok=True)
            torch.save(net.state_dict(), f"{directory}/weights.pt")

            maxAuc = auc
        else :
            print(f"(aborting) the current auc {auc} is smaller than the previous auc {maxAuc}")

if __name__ == "__main__" :
    
    config = parse()
    
    print("configs : ")
    print(f"   mil       : {config.mil}")
    print(f"   pretrain  : {config.pretrain}")    
    print(f"   diagnosis : {config.diagnosis}")
    print(f"   mode      : {config.mode}")
        
    # dataloaders
    ###################################################
    trainGen = DataGen(f"{config.dataPath}/features/{config.pretrain}",
                       f"{config.dataPath}/label.csv",
                       "./data/features.pkl",
                       diag    = config.diagnosis,
                       mode    = config.mode,
                       train   = True,
                       augment = False)
    
    validGen = DataGen(f"{config.dataPath}/features/{config.pretrain}",
                       f"{config.dataPath}/label.csv",
                       "./data/features.pkl",
                       diag    = config.diagnosis,  
                       mode    = config.mode,
                       train   = False,
                       augment = False)
        
    weights = COMPUTE_WEIGHTS(trainGen, weightType = config.weightType)
        
    trainLoader = DataLoader(trainGen,
                             batch_size  = 1,
                             shuffle     = True if config.weightType != "sampler" else None,
                             pin_memory  = True,
                             num_workers = config.cpuN,
                             sampler     = sampler(weights, len(weights)) if config.weightType == "sampler" else None)
        
    validLoader = DataLoader(validGen,
                             batch_size  = 1,
                             shuffle     = False,
                             pin_memory  = False,
                             num_workers = config.cpuN)
    
    
    # model
    ###################################################    
    NET = {False : lambda : CLAM     (in_channels = config.featureN, n_classes = 1, mil = config.mil),
           True  : lambda : ResNet1D (in_channels = config.featureN, n_classes = 1, mil = config.mil, sliceNorm = config.sliceNorm)}
    
    net = NET["3D" in config.mode]()
    
    lossFn    = COMPUTE_LOSS(weights if config.weightType == "loss" else None)
    optimizer = GET_OPTIMIZER(net.parameters(), config.optimizer, config.lr, 0)
        
    # train & valid
    ###################################################
    train(trainLoader, validLoader,
          net.cuda(),
          lossFn,
          optimizer, 
          config)