import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Function
import numpy as np
from binarizers import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = BinarizeConv2d(1, 6, 5)
        #self.conv2 = BinarizeConv2d(6, 16, 5)
        
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        #self.fc1 = BinarizeLinear(256, 120)
        #self.bn1 = nn.BatchNorm1d(120)
        
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        #self.fc2 = BinarizeLinear(120, 84)
        #self.bn2 = nn.BatchNorm1d(84)
        
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
        #self.fc3 = BinarizeLinear(84, 10)
        
        self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
        
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)
        

    def forward(self, x):        
        #x = F.max_pool2d(F.hardtanh(self.conv1(x)), (2, 2))
        #x = F.max_pool2d(F.hardtanh(self.conv2(x)), (2, 2))
                
        #x = x.view(-1, self.num_flat_features(x))
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        #x = F.hardtanh(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        #x = F.hardtanh(x)
        
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        
        x = self.fc4(x)
        
        return self.logsoftmax(x)
        #return x
        
    #def num_flat_features(self, x):
        #return np.prod(x.size()[1:])
        