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


def lower_bound(net):
    decomp = decompose_net(net)
    linear, bn = decomp['linear'], decomp['bn']
    
    D = [[] for _ in range(len(bn))]

    for i in range(len(bn)):
        for j in range(linear[i]['weight'].shape[0]):
            val = ((-bn[i]['std'][j]/bn[i]['scale'][j])*bn[i]['bias'][j] + \
                   bn[i]['mean'][j] - linear[i]['bias'][j]).item()

            if bn[i]['scale'][j] > 0:
                val = ceil(val)
            else:
                val = floor(val)
            
            C_j = val
            C_j_prime = ceil(C_j/2 + linear[i]['weight'][j,:].sum().item()/2)
            D_j = C_j_prime + (linear[i]['weight'][j,:] == -1).sum().item()
            
            D[i].append(D_j)
            
    return D