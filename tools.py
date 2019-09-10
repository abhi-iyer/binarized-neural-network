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
from math import ceil, floor

from tools import *
from encoding import *
from models import *
from bnn import *

def verify_net(net):
    decomp = decompose_net(net)
    linear, bn = decomp['linear'], decomp['bn']
    
    for each in linear:
        weight = ((each['weight'] == -1).sum() + (each['weight'] == 1).sum()).item()
                
        assert weight == each['weight'].numel()
        
def manual_prop(manual, linear, bn):
    layer1 = torch.matmul(linear[0]['weight'], manual) + linear[0]['bias'].unsqueeze(1)
    layer1 = bn[0]['scale'] * ((layer1 - bn[0]['mean']) / bn[0]['std']) + bn[0]['bias']
    layer1 = binarize(layer1)
        
    layer2 = torch.matmul(linear[1]['weight'], layer1) + linear[1]['bias'].unsqueeze(1)
    layer2 = bn[1]['scale'] * ((layer2 - bn[1]['mean']) / bn[1]['std']) + bn[1]['bias']
    layer2 = binarize(layer2)

    layer3 = torch.matmul(linear[2]['weight'], layer2) + linear[2]['bias'].unsqueeze(1)
    layer3 = bn[2]['scale'] * ((layer3 - bn[2]['mean']) / bn[2]['std']) + bn[2]['bias']
    layer3 = binarize(layer3)

    layer4 = torch.matmul(linear[3]['weight'], layer3) + linear[3]['bias'].unsqueeze(1)

    return layer4.squeeze(2).argmax(dim=1).cpu()

def manual_acc(net, loader):
    total = 0
    
    decomp = decompose_net(net)
    linear, bn = decomp['linear'], decomp['bn']

    for (data, target) in loader:
        data = data.cuda()

        manual = data.view(loader.batch_size, -1, 1)
        manual = binarize(manual)
        manual_out = manual_prop(manual, linear, bn)
        
        total += (manual_out == target).sum().item()

    print(total / len(loader.dataset))