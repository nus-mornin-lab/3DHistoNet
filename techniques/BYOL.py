import copy, random

import argparse
from argparse import ArgumentParser

from toolz import *
from toolz.curried import *
from itertools import islice

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torchvision import transforms as T
import torchvision.transforms.functional as TF

from PIL import Image,ImageFilter, ImageOps

class BYOL(nn.Module):
    
    def __init__(
        self, net,
        ZInSize  = 1024,
        ZOutSize = 256,
        ZMidSize = 4096,
        emaR     = 0.99):
        
        super().__init__()

        self.ema = EMA(emaR)        
        self.augment = Augment()
        
        self.net       = net                
        self.student   = nn.Sequential(self.net, MLP(ZInSize, ZOutSize, ZMidSize))        
        self.teacher   = copy.deepcopy(self.student); RequireGrad(self.teacher, False)             
        self.predictor = MLP(ZOutSize, ZOutSize, ZMidSize)
        

    def update_moving_average(self):
        update_moving_average(self.ema,self.teacher, self.student)

    def forward(self, v1, v2):
        
        v1s = self.predictor(self.student(v1))
        v2s = self.predictor(self.student(v2))
        
        with torch.no_grad():
            v1t = self.teacher(v1) ; s1v1T.detach_()
            v2t = self.teacher(v2) ; s1v2T.detach_()
            
        loss1 = mseLoss(v1s, v2t.detach())
        loss2 = mseLoss(v2s, v1t.detach())
        
        loss = loss1 + loss2
        
        return loss.mean()
    
    
    def genTask(self, slice1):
        
        s1v1, s1v2 = self.augment(slice1)
        
        return s1v1, s1v2


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)
    
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)
            
def RequireGrad(model, val):
    for p in model.parameters():
        p.requires_grad = val    
                
def mseLoss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def NCELoss(out_1, out_2, temperature = 0.5, eps=1e-6):
    """
    out_1: [N, C]
    out_2: [N, C]
    """
    
    out_1 = F.normalize(out_1, dim=-1, p=2)
    out_2 = F.normalize(out_2, dim=-1, p=2)

    # gather representations in case of distributed training
    out_1_dist = out_1
    out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss


class Augment(object):
    
    """
    modified from 
        "https://github.com/facebookresearch/dino"
        "https://github.com/lucidrains/byol-pytorch"
    """
    
    def __init__(self, aug1Scale = (0.4, 1.), aug2Scale = (0.4, 1.)):
        
        # student
        self.aug1 = T.Compose([         
            T.RandomResizedCrop(256, scale=aug1Scale),
            T.RandomApply([Rotation(angles=[0, 90, 180, 270])], p = 0.25),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur(0.1)], p=0.1),
            T.RandomSolarize(0.5, p = 0.2),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
        
        # teacher
        self.aug2 = T.Compose([
            T.RandomResizedCrop(256, scale=aug2Scale),
            T.RandomApply([Rotation(angles=[0, 90, 180, 270])], p = 0.25),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur(0.1)], p=0.1),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])

    def __call__(self, image):
        """
        image :: [... , 3, H, W] 
        """
        return self.aug1(image), self.aug2(image)
    
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Rotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

    
if __name__ == '__main__': 

    import sys, os    
    sys.path.append("..")
        
    from easydict import EasyDict as edict

    from data.DataGenSSL import DataGen
    from matplotlib import pyplot as plt
    
    from models.ResNet import resnet50_baseline    
    
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"        

    def normalise(x):
        
        val_max = 1
        val_min = 0
        
        _min = torch.tensor(x.min(dim = 1)[0].min(dim = 1)[0]).unsqueeze(-1).unsqueeze(-1)
        _max = torch.tensor(x.max(dim = 1)[0].max(dim = 1)[0]).unsqueeze(-1).unsqueeze(-1)
        
        y = ( x - _min ) / ( _max - _min )
        
        return y        
      
    net  = resnet50_baseline(pretrained=None).cuda()    
    byol = BYOL(net).cuda()    
   
    gen = DataGen("../data/datasets/processed/flatten", augment = byol.genTask, train = True)

    for i in range(len(gen)):
        v1, v2 = gen.__getitem__(i)

        v1 = normalise(v1)
        v2 = normalise(v2)

        fig, axs = plt.subplots(ncols = 2)
        axs[0].imshow(v1.permute(1,2,0))
        axs[0].set_title('view 1')
        axs[1].imshow(v2.permute(1,2,0))
        axs[1].set_title('view 2')
        fig.suptitle(f'slice 1', fontsize=16)        
        plt.show()
