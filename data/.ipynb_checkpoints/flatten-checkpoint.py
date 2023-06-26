import os, sys, time, warnings, glob, tqdm, h5py

import argparse
from argparse import ArgumentParser

from toolz import *
from toolz.curried import *
from itertools import islice

import numpy as np

def Parse():
    
    parser = ArgumentParser()
    
    parser.add_argument("--dataLoadPath", type=str, default="./datasets/processed/structured")
    parser.add_argument("--dataSavePath", type=str, default="./datasets/processed/flatten")
    parser.add_argument("--nproc", type=int, default=80)
    
    # get args
    args = first(parser.parse_known_args())
    
    return args

@curry
def process(config, file):
    
    h5 = h5py.File(file, "r")
    caseId = file.split("/")[-1].replace(".h5", "")
    
    for k, v in h5.items():
        
        data = v[()]
        print(data.shape)
        H, W, C, D, PW, PH = data.shape

        for h in range(H):
            for w in range(W):
                saveDir = f"{config.dataSavePath}/{caseId}|{k}|{h}|{w}"
                os.mkdir(saveDir)                    
                for d in range(D):
                    np.save(f"{saveDir}/{d}.npy", data[h,w,:,d, ...])
                           
if __name__ == '__main__':
    
    from multiprocessing import Pool
        
    config = Parse()
    
    files = glob.glob(f"{config.dataLoadPath}/*.h5")
    
    with Pool(80) as p:
        p.map(process(config), files)
