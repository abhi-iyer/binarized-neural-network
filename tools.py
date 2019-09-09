import torch
import torch.nn as nn
from encoding import *

def verify_net(net):
    decomp = decompose_net(net)
    linear, bn = decomp['linear'], decomp['bn']
    
    for each in linear:
        weight = ((each['weight'] == -1).sum() + (each['weight'] == 1).sum()).item()
        bias = ((each['bias'] == -1).sum() + (each['bias'] == 1).sum()).item()
                
        assert weight == each['weight'].numel()
        assert bias == each['bias'].numel()
        
def manual_prop(manual):
    layer1 = torch.matmul(linear[0].weight, manual) + linear[0].bias.unsqueeze(1)
    layer1 = bn[0][0] * ((layer1 - bn[0][2]) / bn[0][3]) + bn[0][1]
    layer1 = binarize(layer1)
        
    layer2 = torch.matmul(linear[1].weight, layer1) + linear[1].bias.unsqueeze(1)
    layer2 = bn[1][0] * ((layer2 - bn[1][2]) / bn[1][3]) + bn[1][1]
    layer2 = binarize(layer2)

    layer3 = torch.matmul(linear[2].weight, layer2) + linear[2].bias.unsqueeze(1)
    layer3 = bn[2][0] * ((layer3 - bn[2][2]) / bn[2][3]) + bn[2][1]
    layer3 = binarize(layer3)

    layer4 = torch.matmul(linear[3].weight, layer3) + linear[3].bias.unsqueeze(1)

    return layer4.squeeze(2).argmax(dim=1).cpu()

def manual_acc(loader):
    total = 0

    for (data, target) in loader:
        data = data.cuda()

        manual = data.view(loader.batch_size, -1, 1)
        manual = binarize(manual)
        manual_out = manual_prop(manual)
        
        total += (manual_out == target).sum().item()

    print(total / len(loader.dataset))