import os, sys, time, warnings, glob, tqdm, h5py

warnings.filterwarnings('ignore')

import torch.utils.model_zoo as model_zoo


import argparse
from argparse import ArgumentParser

from toolz import *
from toolz.curried import *
from itertools import islice

import torch
from torch import nn, optim
import torchvision
import torch.nn.functional as F
import numpy as np

from skimage.filters import gaussian
from matplotlib import pyplot as plt

from models.ResNet import resnet50_baseline, load_pretrained_weights

def Parse():
    
    parser = ArgumentParser()
    
    parser.add_argument("--dataLoadPath", type=str, default="./data/datasets/processed/structured")
    parser.add_argument("--dataSavePath", type=str, default="./data/datasets/features/SIMCLR")    
    parser.add_argument("--ckptLoadPath", type=str, default="./data/datasets/features/SIMCLR")    

    # for IMAGENET
    #parser.add_argument("--dataSavePath", type=str, default="./data/datasets/features/IMAGENET")        
    #parser.add_argument("--ckptLoadPath", type=str, default="https://download.pytorch.org/models/resnet50-19c8e357.pth")
    
    parser.add_argument("--gpuN", type=str, default= "0")
    parser.add_argument("--numWorkers", type=int, default=6)
    
    # get args
    args = first(parser.parse_known_args())
    
    # pick a gpu that has the largest space
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuN
        
    return args


def DataGen(dataLoadPath):
    
    def normalize(image):
        
        """
        image :: [PH,PW,3,D,256,256]
        """
        
        image = image/255.

        sourceMean = torch.tensor((0.5,0.5,0.5)).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sourceStd  = torch.tensor((0.5,0.5,0.5)).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # for IMAGENET
#         sourceMean = torch.tensor([0.856, 0.692, 0.823]).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         sourceStd  = torch.tensor([0.108, 0.171, 0.110]).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        image = (image-sourceMean)/sourceStd
        
        return image

    files = glob.glob(f"{dataLoadPath}/*.h5")
    
    for file in files:
        
        h5 = h5py.File(file, "r")
        
        caseId = file.split("/")[-1].replace(".h5", "")
        data   = {k : normalize(torch.tensor(h5[k][()]).type(torch.float32)) for k in h5.keys()} 
        
        yield caseId, data
        
        
def extractFeature(model, data, dataSavePath) :
    
    """
    image    :: { key : [PH,PW,3,D,256,256] }
    features :: { key : [PH,PW,D,C'] }
    dataSavePath :: str
    """
    
    features = {}
    for key, slices in data.items()  :
        
        print(f"    for {key}")
        PH,PW,C,D,H,W = slices.shape
        slices = slices.permute(3,0,1,2,4,5).view(D,PH*PW,C,H,W)
        
        feature = []
        
        # a standard way
        if "F" not in dataSavePath:
            for _slice in slices:            
                _slice = _slice.to("cuda")
                feature.append(model(_slice).view(PH,PW,-1).detach().cpu().numpy())            
                
        # a fake way
        if "F" in dataSavePath:
            if D > 7:
                slices = torch.stack([slices[7, ...]]*17, dim = 0)
            else:
                slices = torch.stack([slices[0, ...]]*17, dim = 0)
                
            for i, _slice in enumerate(slices):
                
                kSize = 2*abs(i-8)+1
                
                if kSize != 1:
                    _slice = torchvision.transforms.functional.gaussian_blur(_slice,kSize,[6,6])         
                else:
                    _slice = _slice
                    
#                 plt.imshow(_slice[0].permute(1,2,0))
#                 plt.show()
            
                _slice = _slice.to("cuda")                
                feature.append(model(_slice).view(PH,PW,-1).detach().cpu().numpy())
                                
        features[key] = np.stack(feature)
        
    return features

def computeMeanStd(dataGen, n = 50):
    
    globalMean = []
    globalStd  = []
    for _, data in tqdm.tqdm(islice(dataGen, n)):
        
        for _, slices in data.items()  :
            
            localMean = torch.mean(slices, axis = (0,1,3,4,5))
            localStd  = torch.std(slices, axis = (0,1,3,4,5))
            
            globalMean.append(localMean)
            globalStd.append(localStd)
            
    globalMean = sum(globalMean)/len(globalMean)
    globalStd  = sum(globalStd)/len(globalStd)
    
    print("globalMean : ", globalMean)
    print("globalStd  : ", globalStd)

if __name__ == '__main__':
        
    configs = Parse()    
    
    dataGen = DataGen(configs.dataLoadPath)

    temp = torch.load(configs.ckptLoadPath)    
    model = resnet50_baseline(pretrained = configs.ckptLoadPath).eval().to("cuda")
    
    for caseId, data in tqdm.tqdm(dataGen) :
        
        if not os.path.exists(f"{configs.dataSavePath}/{caseId}.pt"):
        
            print(f"extracting feature from patient No : {caseId}")
            
            features = extractFeature(model, data, configs.dataSavePath)
            
            torch.save(features, f"{configs.dataSavePath}/{caseId}.pt")
            
        else:
            print(f"{caseId} done alr")