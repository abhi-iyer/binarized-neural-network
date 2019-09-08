import torch
import torch.nn as nn
import torch.nn.init as init
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
from binarizers import *

class MNIST_BNN(nn.Module):
    def __init__(self):
        super(MNIST_BNN, self).__init__()      

        self.fc1 = BinarizeLinear(784, 784*2)
#         self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(784*2, eps=0)
#         init.normal_(self.bn1.weight.data, mean=1, std=0.02)
#         init.constant_(self.bn1.bias.data, 0)
        
        self.fc2 = BinarizeLinear(784*2, 784*2)
#         self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(784*2, eps=0)
#         init.normal_(self.bn2.weight.data, mean=1, std=0.02)
#         init.constant_(self.bn2.bias.data, 0)
        
        self.fc3 = BinarizeLinear(784*2, 784*2)
#         self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(784*2, eps=0)
#         init.normal_(self.bn3.weight.data, mean=1, std=0.02)
#         init.constant_(self.bn3.bias.data, 0)
        
        self.fc4 = BinarizeLinear(784*2, 10)
        
#         self.logsoftmax = nn.LogSoftmax(dim=1)
            
    def forward(self, x):                        
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
#         x = self.htanh1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
#         x = self.htanh2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
#         x = self.htanh3(x)
        
        x = self.fc4(x)
        
        return x
                
#         return self.logsoftmax(x)
#         return self.softmax(x)
        