"""
Simplified from https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py

We use the default setting in the code:
    1. dropOut is deactivated
    2. no clustering loss
    3. small neural net version.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLAM(nn.Module):
    def __init__(self, in_channels = 1024, n_classes = 1, mil = "attention"):
        super(CLAM, self).__init__()

        self.in_channels = in_channels
        self.n_classes   = n_classes
        self.mil         = mil 
    
        self.head = nn.Sequential(nn.Linear(in_channels, in_channels//2), nn.ReLU(),
                                  nn.Linear(in_channels//2, in_channels//2), nn.ReLU())
        
        if self.mil == "attention":            
            self.attention = Attn_Net_Gated(L = in_channels//2,
                                            D = in_channels//4,
                                            n_classes = n_classes)
        
        
        self.classifiers = nn.Linear(in_channels//2, n_classes)
        
    def forward(self, x):
        
        x = x[:,:,0]
        
        z = self.head(x)
        
        if self.mil == "attention":            
            A, z = self.attention(z)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN        
            A_raw = A        
            A = F.softmax(A, dim=1)  # softmax over N        
            z = torch.mm(A, z)
            
        if self.mil == "mean":            
            z = z.mean(0).unsqueeze(0)
            
        if self.mil == "max":
            z = z.max(dim=0)[0].unsqueeze(0)
                    
        logits = self.classifiers(z)
        
        return (logits, z, A_raw if self.mil == "attention" else None)
    
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        
        a = self.attention_a(x)
        b = self.attention_b(x)
        
        A = self.attention_c(a.mul(b))  # N x n_classes
        
        return A, x
    

if __name__ == '__main__':
    
    X = torch.randn(200, 1024, 16).cuda()
    
    net = CLAM(in_channels = 1024, n_classes = 1, mil = "attention").cuda()
    
    logit, A = net(X)
    
    print(logit.shape)
    