import os, sys, glob, csv, random, re, operator

from toolz import *
from toolz.curried import *

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

from PIL import Image
#########################################################################################################
class DataGen(Dataset):
        
    def __init__(self, dataDir, transform = identity,
                 nSlice = 1, is2D = False, train = True) :

        self.dataDir   = dataDir
        self.train     = train
        self.transform = transform
        self.nSlice    = nSlice 
        self.is2D      = is2D
        self.folders   = parse(dataDir, train)
        
    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, i):
        
        folder = self.folders[i]
        
        if self.is2D == False:
            
            if len(glob.glob(f"{folder}/*")) >= 2:                
                sliceNs = sorted(random.sample(range(len(glob.glob(f"{folder}/*"))), self.nSlice))            
            elif len(glob.glob(f"{folder}/*")) < 2:                
                sliceNs = [0,0]
                
        else:
            sliceNs = [7] # middle
            
        if self.nSlice != 1:
            dif = (sliceNs[1] - sliceNs[0]) - 1
        else :
            dif = None
        
        patches = [np.load(f"{folder}/{n}.npy") for n in sliceNs]
            
        patch  = np.stack(patches, axis = 1)
                    
        # patch : [3,D,256,256]
        patch = torch.tensor(patch)
        
        return self.transform( (patch, dif) )
#########################################################################################################    
def parse(dataDir, train):
    return #add your own data parser here

if __name__ == '__main__':
    
    from easydict import EasyDict
    from matplotlib import pyplot as plt
        
    def normalise(x):
        
        val_max = 1
        val_min = 0
        
        _min = torch.tensor(x.min(dim = 1)[0].min(dim = 1)[0]).unsqueeze(-1).unsqueeze(-1)
        _max = torch.tensor(x.max(dim = 1)[0].max(dim = 1)[0]).unsqueeze(-1).unsqueeze(-1)
        
        y = ( x - _min ) / ( _max - _min )
        
        return y        
    
    D = 16
    config = EasyDict()
    config.dataDir = "./datasets/processed/flatten"
    config.train   = True
    config.nSlice  = 1
    config.is2D    = True 
    
    gen = DataGen(config.dataDir,
                  nSlice = config.nSlice,
                  is2D   = config.is2D,
                  train  = config.train)
    
    for i in range(len(gen)):        
        
        volume = gen.__getitem__(i)[0]
        fig, axs = plt.subplots(nrows = 1, ncols = config.nSlice, figsize=(30,30*config.nSlice))
        
        for i, ax in enumerate([axs]):            
            slice = normalise(volume[:, i, ...])            
            ax.imshow(slice.permute(1,2,0))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
                        
            
            
            
            