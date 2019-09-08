import torch
import torch.nn as nn

def decompose_net(net):
    linear = []
    bn = []

    for name, module in list(net._modules.items()):
        if 'bn' in name:
            scale = torch.cuda.FloatTensor(module.weight).unsqueeze(dim=1)
            bias = torch.cuda.FloatTensor(module.bias).unsqueeze(dim=1)
            mean = torch.cuda.FloatTensor(module.running_mean).unsqueeze(dim=1)
            std = torch.cuda.FloatTensor(module.running_var ** 0.5).unsqueeze(dim=1)

            bn.append((scale, bias, mean, std))

        if 'fc' in name:
            linear.append(module)
            
    return {"linear" : linear, "bn" : bn}