import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Function
import numpy as np

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
        binary_weight = binarize(self.weight)
        
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)
   
    def reset_parameters(self):
        in_features, out_features = self.weight.size()
        
        stdv = math.sqrt(1.5 / (in_features + out_features)) 
        self.weight.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            self.bias.data.zero_()
            
        self.weight.lr_scale = 1. / stdv

        
class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        binary_weight = binarize(self.weight)
        
        return F.conv2d(input, binary_weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
    
    def reset_parameters(self):
        in_features, out_features = self.in_channels, self.out_channels
        
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        
        stdv = math.sqrt(1.5 / (in_features + out_features)) 
        self.weight.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            self.bias.data.zero_()
            
        self.weight.lr_scale = 1. / stdv
        

        
            
            
            

