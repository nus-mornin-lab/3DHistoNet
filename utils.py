import os, sys

import pandas as pd
import numpy as np

from toolz import *
from toolz.curried import *

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from sklearn.metrics import roc_curve, precision_recall_curve, auc as get_auc
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop

def get_rank():
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def init_distributed_mode(args):
        
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, "env://"), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)
    
    
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    
def hasBN(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def GET_OPTIMIZER(params, opt_name, learning_rate, weight_decay):
    
    return \
        {"SGD"      : lambda : SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay),
         "Adadelta" : lambda : Adadelta(params, lr=learning_rate, weight_decay=weight_decay),
         "Adagrad"  : lambda : Adagrad(params, lr=learning_rate, weight_decay=weight_decay),
         "Adam"     : lambda : Adam(params, lr=learning_rate, weight_decay = weight_decay),
         "RMSprop"  : lambda : RMSprop(params, lr=learning_rate, momentum=0.9)}[opt_name]()

def COMPUTE_METRIC(outputs, targets):
     
    fpr, tpr, _ = roc_curve(targets, outputs, drop_intermediate = False)

    auc = get_auc(fpr, tpr)
        
    aucs = float(str(auc)[:5])
    fprs = np.linspace(0, 1, 100)    
    tprs = np.interp(np.linspace(0, 1, 100), fpr, tpr); tprs[-1] = 1.0
    
    return {'aucs' : aucs,
            'fprs' : fprs,
            'tprs' : tprs}

@curry
def COMPUTE_LOSS(weight, output, target):    
     return F.binary_cross_entropy_with_logits(output.squeeze(),
                                               target.squeeze().type_as(output),
                                               pos_weight = weight.cuda() if weight != None else None)
    
@curry
def COMPUTE_WEIGHTS(dataset, weightType = "sampler"):
    
    targets = compose(torch.tensor, list, map(second))(dataset.datapoints)
    
    if weightType == "sampler" : 
        return (1/np.unique(targets, return_counts=True)[1])[targets]
    
    if  weightType == "loss" : 
        return (len(targets) - sum(targets)) / sum(targets)

if __name__ == '__main__':
            
    # test weights
    
    output = torch.sigmoid(torch.rand(10,4)-0.5).to("cuda")
    target = torch.randint(0,2, [10,4]).to("cuda")
    weights = COMPUTE_WEIGHTS("./data/datasets/label.csv")    
    batch_weight = True
        
    print(COMPUTE_LOSS(output, target, batch_weight, weights))
    print(COMPUTE_METRIC(output.detach().cpu().numpy(), target.detach().cpu().numpy()))
                

