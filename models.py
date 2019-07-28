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
from binarizers import *

class MNIST_BNN(nn.Module):
    def __init__(self):
        super(MNIST_BNN, self).__init__()      
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
        
        self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.drop = nn.Dropout(0.5)
        

    def forward(self, x):                        
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        
        x = self.fc4(x)
                
        return self.logsoftmax(x)
        
    def flatten(self, x):
        return np.prod(x.size()[1:])
        