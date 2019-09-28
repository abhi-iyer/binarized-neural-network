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
    
    C = [[] for _ in range(len(bn))]
    D = [[] for _ in range(len(bn))]

    for i in range(len(bn)):
        for j in range(len(bn[i]['scale'])):
            val = ((-bn[i]['std'][j]/bn[i]['scale'][j])*bn[i]['bias'][j] + \
                   bn[i]['mean'][j] - linear[i]['bias'][j]).item()

            if val > 0:
                val = ceil(val)
            else:
                val = floor(val)

            C[i].append(val)

    for i in range(len(bn)):
        for j in range(len(linear[i]['weight'])):
            C_prime = ceil(C[i][j]/2 + linear[i]['weight'][j, :].sum().item()/2)
            val = C_prime + (linear[i]['weight'][j, :] == -1).sum().item()

            D[i].append(val)
            
    return D