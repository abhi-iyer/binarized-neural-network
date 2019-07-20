import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import os

path = "~/binarized-neural-network/"

def create_dataloader(name):
    if name == "MNIST":
        directory = path + "mnist/"
        
        train = datasets.MNIST(root=directory, train=True, download=True, 
                               transform=transforms.Compose([transforms.ToTensor(), 
                                                             transforms.Normalize((0.1307,), (0.3081,))]))
        
        test = datasets.MNIST(root=directory, train=False,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))]))
        
        train_loader = DataLoader(train, batch_size=64, shuffle=True, pin_memory=True, num_workers=1)
        test_loader = DataLoader(test, batch_size=1000, shuffle=True, pin_memory=True, num_workers=1)
                                 
    return train_loader, test_loader

train_loader, test_loader = create_dataloader('MNIST')