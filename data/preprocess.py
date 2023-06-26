"""
기모찌
"""

import time, os, sys, glob
import h5py

from tqdm import tqdm

from toolz import * 
from toolz.itertoolz import *
from toolz.curried import * 
from toolz.curried.operator import * 
from itertools import islice

import torch
from torch.nn.functional import pad

import numpy as np

import staintools
standardize = staintools.LuminosityStandardizer.standardize

from skimage import io
from matplotlib import pyplot as plt

def _normalize(trainer, _query):
    
    """
    _query          :: np.array[H,W,3]
    _queryTransform :: np.array[H,W,3]
    
    https://github.com/Peter554/StainTools
    """
    
    _queryTransform = compose(trainer.transform,
                              standardize)(_query)
    
    return _queryTransform

def normalize(trainer, query):
    
    """    
    query          :: np.array[D,H,W,3]    
    queryTransform :: np.array[D,H,W,3]
    """
    
    queryTransform = np.empty(query.shape)
    for i, querySide in enumerate(query):
        queryTransform[i] = _normalize(trainer, querySide)
        
    return queryTransform

def Pachify(slide, patchSize=256):
    
    """
    slide :: np.array[D,H,W,3]
    """
    
    slide = torch.tensor(slide)
    
    N,H,W,C = slide.shape
    
    if H < patchSize :
        padN = patchSize - H
        slide = pad(slide, ((0,0)+(0,0)+(padN,0)+(0,0)), "constant", 0)        
    if W < patchSize :
        padN = patchSize - W
        slide = pad(slide, ((0,0)+(padN,0)+(0,0)+(0,0)), "constant", 0)        
        
    N,H,W,C = slide.shape
    
    paches = (slide
                .unfold(1, size=patchSize, step=patchSize)
                .unfold(2, size=patchSize, step=patchSize)
                .permute(1,2,3,0,4,5)
                .reshape(H//patchSize,W//patchSize,C,N,patchSize,patchSize))
    
    return np.array(paches)
    
@curry
def process(trainer, queryPath, saveDir = "./datasets/processed", unit = np.uint8):
    
    """
    queryPath  :: a path to query  patient        
    """
    
    print(f"processing {queryPath} ... ")
    
    caseId = queryPath.split("/")[-1]
    
    hf = h5py.File(f"{saveDir}/{caseId}.h5", 'w')

    for querySlidePath in glob.glob(f"{queryPath}/*.tif"):

        normalized = normalize(trainer, io.imread(querySlidePath))
        #normalized = io.imread(querySlidePath)

        patchfied = Pachify(normalized).astype(unit)

        slideId = querySlidePath.split("/")[-1].replace(".tif", "")            
        hf.create_dataset(slideId, data = patchfied)

    hf.close()
            
# 2. patchify
###########################

if __name__ == "__main__":
    
    from multiprocessing import Pool

    patientPaths = glob.glob("/space/jacob/HISTO/*/*")
    trainer      = staintools.StainNormalizer(method = "vahadane")

    targetPath, *queryPaths = patientPaths
    
    compose(trainer.fit,
            standardize,
            first, io.imread,
            first, glob.glob)(f"{targetPath}/*.tif")
            
    with Pool(8) as p:
        p.map(process(trainer), queryPaths)