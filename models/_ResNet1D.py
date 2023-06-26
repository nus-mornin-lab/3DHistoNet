"""
resnet for 1-d signal data, pytorch version
 
Shenda Hong, Oct 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv1dPadSame(nn.Module):
    
    """
    extend nn.Conv1d to support SAME padding
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out = out + identity

        return out
    
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        
        return A, x
    
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels,
                 base_filters       = 512,
                 kernel_size        = 3,
                 stride             = 1,
                 groups             = 1,
                 n_block            = 2,
                 n_classes          = 4,
                 downsample_gap     = 1,
                 mil                = "mean",
                 k_sample           = 8,
                 use_bn  = False,
                 use_do  = False,
                 verbose = False):
        super(ResNet1D, self).__init__()
        
        self.k_sample    = k_sample
        self.mil         = mil
        self.verbose     = verbose
        self.n_block     = n_block
        self.kernel_size = kernel_size
        self.stride      = stride
        self.groups      = groups
        self.use_bn      = use_bn
        self.use_do      = use_do

        self.downsample_gap = downsample_gap # 2 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                in_channels  = out_channels
                out_channels = int(in_channels * 0.5)             
                
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            
            self.basicblock_list.append(tmp_block)
        
        if mil == "attention" :
            self.att = Attn_Net_Gated(L = out_channels, D = out_channels, n_classes = n_classes)
            self.instance_classifier = nn.Linear(out_channels, 2)
            self.instance_loss_fn = nn.CrossEntropyLoss()
            
        self.dense = nn.Linear(out_channels, n_classes)
        
    @staticmethod
    def create_positive_targets(length):
        return torch.full((length, ), 1).cuda().long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0).cuda().long()        
        
    def inst_eval(self, A, h, classifier):
        
        if len(A.shape) == 1:
            A = A.view(1, -1)
            
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        
        p_targets = self.create_positive_targets(self.k_sample)
        n_targets = self.create_negative_targets(self.k_sample)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        
        return instance_loss, all_preds, all_targets
        
    def forward(self, x):
        
        out = x
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose: print(out.shape)
        
        if self.mil == "mean":            
            out = out.mean(-1).mean(0).unsqueeze(0)            
            return self.dense(out), None
            
        if self.mil == "max":            
            out = out.mean(-1).max(dim=0)[0].unsqueeze(0)
            return self.dense(out), None
            
        if self.mil == "attention":
            out = out.mean(-1)
            A, out = self.att(out)
            A = F.sigmoid(A)  # softmax over N
            out = A * out
            return self.dense(out.max(dim = 0)[0].unsqueeze(0)), None
            
#             A = torch.transpose(A, 1, 0)[0]  # KxN            
#             A = F.softmax(A, dim=1)  # softmax over N
            
#             if y.item() == 1: #in-the-class:
#                 instance_loss, *_ = self.inst_eval(A, out, self.instance_classifier)
    
if __name__ == "__main__":
    
    XShape = (10, 1024, 17)
    X = torch.randn(*XShape).type(torch.float32)
    X = X.cuda()
    
    net = ResNet1D(XShape[1],
                   base_filters       = 512,
                   kernel_size        = 3,
                   stride             = 1,
                   groups             = 1,
                   n_block            = 5,
                   n_classes          = 1,
                   downsample_gap     = 1,
                   mil                = "attention",
                   k_sample           = 8).cuda()
    
    logit = net(X)
        
    

