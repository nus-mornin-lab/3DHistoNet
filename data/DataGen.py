import os, sys, glob, csv, random, re, operator, pickle

from toolz import *
from toolz.curried import *

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

from matplotlib import pyplot as plt

######################################################################################################### 
class DataGen(Dataset):
        
    def __init__(self, dataDir, labelPath,
                 train   = True,
                 diag    = "ER",
                 mode    = "3D") :

        self.mode      = mode
        self.dataDir   = dataDir
        self.labelPath = labelPath
        self.train     = train
        self.diag      = diag
        
        self.datapoints = self._parse()            
        
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, i):
        
        patches, labels, caseId, patchIds = self.__getitemRaw__(i)
        
        return patches, labels, caseId, patchIds
    
    def __getitemRaw__(self, i):
        
        dataPath, *labels, _ = self.datapoints[i]
        
        data = torch.load(dataPath)
        
        patchIds, _patches = data.keys(), list(data.values())
        
        patches = []
        for _patch, patchId in zip(_patches, patchIds):
            
            D,H,W,C = _patch.shape

            # [D,H,W,C] -> [H,W,D,C] -> [H*W,C,D]
            _patch = torch.tensor(_patch).permute(1,2,-1,0).view(H*W,C,D)
            
            if "DCONV" in self.dataDir :                
                if D != 16 :                    
                    _patch = torch.nn.functional.interpolate(_patch, size = [16])
            else:
                if D != 17 :
                    _patch = torch.nn.functional.interpolate(_patch, size = [17])
                    
            patches.append(_patch)
        try:
            _patches = torch.cat(patches).type(torch.float)
            labels   = torch.tensor(labels).type(torch.float)
            caseId   = dataPath.split("/")[-1].split(".")[0]
        except:
            print(dataPath)
        
        if self.mode in ["3D", "3DF"] :
            patches = _patches
        
        if self.mode == "3D2" :            
            idx = list(map(lambda i : (2*i)+1, range(16//2)))
            patches = _patches[:,:,idx]

        if self.mode == "3D4" :
            idx = list(map(lambda i : (4*i)+1, range(16//4)))
            patches = _patches[:,:,idx]
            
        if self.mode == "2D" :
            patches = _patches[:,:,7].unsqueeze(-1)
            
        return (patches, labels, caseId, list(patchIds))        
    
    def _parse(self):
        return #add your own data parser here
    
#########################################################################################################
        
if __name__ == '__main__':
    
    from easydict import EasyDict
    from operator import eq
    
    config = EasyDict()    
    config.dataDir   = "./datasets/features/SIMCLR" 
    config.labelPath = "./datasets/label.csv"               
    config.diag      = "ER"
    config.mode      = "3D"
    train            = True
    agument          = False
    
    gen1 = DataGen(config.dataDir, config.labelPath,
                   train   = train,
                   diag    = config.diag,
                   mode    = config.mode)
    
    config = EasyDict()    
    config.dataDir   = "./datasets/features/SIMCLRDCONV" 
    config.labelPath = "./datasets/label.csv"           
    config.diag      = "ER"
    config.mode      = "3D"
    train            = True
    agument          = False
    
    gen2 = DataGen(config.dataDir, config.labelPath,
                   train   = train,
                   diag    = config.diag,
                   mode    = config.mode)    
    
    # check if the order of files is matching across diff instances.
    one = compose(list, map(lambda x : x.split("/")[-1]), map(first))(gen1.datapoints)
    two = compose(list, map(lambda x : x.split("/")[-1]), map(first))(gen2.datapoints)
        
    for i in range(len(gen2)):
        
        patches, labels, caseId, patchIds = gen2.__getitem__(i)
        print(patches.shape)
