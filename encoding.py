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

            bn.append({'scale' : scale,
                       'bias' : bias,
                       'mean' : mean,
                       'std' : std})

        if 'fc' in name:
            linear.append({'weight' : module.weight,
                           'bias' : module.bias})
            
    return {"linear" : linear, "bn" : bn}


def lower_bound():
    decomp = decompose_net(bnn.net)
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