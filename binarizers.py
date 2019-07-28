import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Function
import numpy as np
import os
from torchvision import datasets as datasets
from torchvision import transforms as transforms
from torch.utils.data import DataLoader as DataLoader

class Binarize(Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        
        return output
   
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

binarize = Binarize.apply
    
class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        
    def forward(self, input):
        input.data = binarize(input.data)
        
        if not hasattr(self.weight, 'full_precision'):
            self.weight.full_precision = self.weight.data.clone()
        
        self.weight.data = binarize(self.weight.full_precision)
        
        out = F.linear(input, self.weight)
        
        if not self.bias is None:
            self.bias.full_precision = self.bias.data.clone() 
            out += self.bias.view(1, -1).expand_as(out)
           
        return out
    
    def reset_parameters(self):
        in_features, out_features = self.weight.size()
        
        stdev = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdev, stdev)
        
        if self.bias is not None:
            self.bias.data.zero_()
            
        self.weight.lr_scale = 1. / stdev
        
        
class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        
    def forward(self, input):
        input.data = binarize(input.data)
        
        if not hasattr(self.weight, 'full_precision'):
            self.weight.full_precision = self.weight.data.clone()
            
        self.weight.data = binarize(self.weight.full_precision)
        
        out = F.conv2d(input, self.weight, None, self.stride, self.padding, 
                       self.dilation, self.groups)
        
        if not self.bias is None:
            self.bias.full_precision = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
            
        return out
    
    def reset_parameters(self):
        in_features, out_features = self.in_channels, self.out_channels
        
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
            
        stdev = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdev, stdev)
        
        if self.bias is not None:
            self.bias.data.zero_()
            
        self.weight.lr_scale = 1. / stdev