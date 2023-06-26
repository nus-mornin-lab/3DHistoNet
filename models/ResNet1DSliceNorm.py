import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv1dPadSame(nn.Module):
    
    """
    extend nn.Conv1d to support SAME padding
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride        
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size,
            stride=self.stride)

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
            
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        
        # the first conv
        self.relu1 = nn.ReLU()
        
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride)
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu1(out)
        
        return out
    
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
                 base_filters       = 1024,
                 kernel_size        = 3,
                 stride             = 1,
                 n_classes          = 1,
                 mil                = "mean",
                 sliceNorm          = True):
        
        super(ResNet1D, self).__init__()
        
        self.mil         = mil
        self.kernel_size = kernel_size
        self.stride      = stride

        self.head = nn.Sequential(BasicBlock(base_filters,    base_filters//2, 3, 1),
                                  nn.GroupNorm(1, base_filters//2)
                                  BasicBlock(base_filters//2, base_filters//2, 3, 1),
                                  nn.GroupNorm(1, base_filters//2))
        
        if self.mil == "attention":
            self.attention = Attn_Net_Gated(L = in_channels//2,
                                            D = in_channels//4,
                                            n_classes = n_classes)
        
        
        self.classifiers = nn.Linear(in_channels//2, n_classes)
        
    def forward(self, x):
                                  
        out = self.head(x)
        
        if self.mil == "mean":            
            out = out.mean(-1).mean(0).unsqueeze(0)            
            return self.classifiers(out), out, None
            
        if self.mil == "max":            
            out = out.mean(-1).max(dim=0)[0].unsqueeze(0)
            return self.classifiers(out), out, None
            
        if self.mil == "attention":            
            A, out = self.attention(out.mean(-1))  # NxK
            #A, out = self.attention(out.max(-1)[0])  # NxK
            A = torch.transpose(A, 1, 0)  # KxN        
            A_raw = A        
            A = F.softmax(A, dim=1)  # softmax over N        
            out = torch.mm(A, out)
            return (self.classifiers(out), out, A_raw)

    
if __name__ == "__main__":
    
    XShape = (16, 1024, 17)
    X = torch.randn(*XShape).type(torch.float32)
    X = X.cuda()
    
    net = ResNet1D(XShape[1],
                   base_filters       = 1024,
                   kernel_size        = 3,
                   stride             = 1,
                   n_classes          = 1,
                   mil                = "attention").cuda()
    
    logit, A = net(X)
        
    

