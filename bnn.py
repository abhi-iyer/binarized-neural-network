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

from models import *

class BNN():
    def create_dataloader(self, name, train_batch, test_batch):
        path = "~/binarized-neural-network/"

        if name == "MNIST":
            directory = path + "mnist/"

            train = datasets.MNIST(root=directory, train=True, download=True, 
                                   transform=transforms.Compose([transforms.ToTensor(), 
                                                                 transforms.Normalize((0.1307,), (0.3081,))]))

            test = datasets.MNIST(root=directory, train=False,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,), (0.3081,))]))

            train_loader = DataLoader(train, batch_size=train_batch, 
                                      shuffle=True, pin_memory=True, num_workers=1)
            test_loader = DataLoader(test, batch_size=test_batch,
                                     shuffle=True, pin_memory=True, num_workers=1)

        return train, test, train_loader, test_loader
    
    def create_model(self, name):
        if name == "MNIST":
            net = MNIST_BNN().to(self.device)
        
        return net

    def __init__(self, name, output_dir, train_batch=100, test_batch=100, num_epochs=10, lr=1e-3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.train_set, self.test_set, self.train_loader, self.test_loader = self.create_dataloader(name, 
                                                                                                    train_batch, 
                                                                                                    test_batch)
        self.net = self.create_model(name)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        self.epochs = num_epochs
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.num_epochs = num_epochs
        self.lr = lr
        
        self.history = []
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []
        
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(output_dir, 
                                       "checkpoint.pth.tar")
        self.config_path = os.path.join(output_dir, "config.txt")
        
        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)
        
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()
            
    @property
    def epoch(self):
        return len(self.history)
    
    def setting(self):
        return {'Model': self.net,
                'Optimizer': self.optimizer,
                'TrainSet' : self.train_set,
                'TestSet' : self.test_set,
                'TrainBatch': self.train_batch,
                'TestBatch' : self.test_batch}
    
    def __repr__(self):
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string
    
    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'Model': self.net.state_dict(),
                'Optimizer': self.optimizer.state_dict(),
                'History': self.history,
                'TrainLoss' : self.train_loss,
                'TrainAcc' : self.train_acc,
                'TestLoss' : self.test_loss,
                'TestAcc' : self.test_acc}
    
    def load_state_dict(self, checkpoint):
        # load from pickled checkpoint
        self.net.load_state_dict(checkpoint['Model'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        self.history = checkpoint['History']
        self.train_loss = checkpoint['TrainLoss']
        self.train_acc = checkpoint['TrainAcc']
        self.test_loss = checkpoint['TestLoss']
        self.test_acc = checkpoint['TestAcc']
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
    def save(self):
        ''''Saves the experiment on disk, i.e, create/update the last checkpoint.'''        
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)  
    
    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint
    
    def evaluate(self):
        self.net.eval()
        
        loss, correct = 0.0, 0.0

        for data, target in self.test_loader:
            if self.device == 'cuda':
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            output = self.net(data)

            loss += self.criterion(output, target).item()

            pred = torch.max(output, dim=1)[1]

            correct += (pred == target).sum()

        loss = float(loss) / len(self.test_loader.dataset)
        acc = float(correct) / len(self.test_loader.dataset)
        
        return loss, acc
    
    def train(self):
        self.net.train()
        
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        
        for epoch in range(start_epoch, self.epochs):
            
            running_loss, running_acc = 0.0, 0.0
            
            for idx, (data, target) in enumerate(self.train_loader):
                if self.device == 'cuda':
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                self.optimizer.zero_grad()
                output = self.net(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()

                for p in list(self.net.parameters()):
                    if hasattr(p, 'full_precision'):
                        p.data.copy_(p.full_precision)

                self.optimizer.step()

                for p in list(self.net.parameters()):
                    if hasattr(p, 'full_precision'):
                        p.full_precision.copy_(p.data.clamp_(-1,1))

                with torch.no_grad():
                    running_loss += loss.item()
                    pred = torch.max(output, dim=1)[1]
                    running_acc += (pred == target).sum()
                                        
                    if idx % 63 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, idx * len(data), len(self.train_loader.dataset),
                            100. * idx / len(self.train_loader), loss.item()))
                        
                torch.cuda.empty_cache()
        
            train_loss = float(running_loss) / len(self.train_loader.dataset)
            train_acc = float(running_acc) / len(self.train_loader.dataset)
            test_loss, test_acc = self.evaluate()

            print('\nTrain set: Average Loss: {:.4f}, Accuracy: {:.0f}%'.format(train_loss, 100. * train_acc))
            print('Test set: Average Loss: {:.4f}, Accuracy: {:.0f}%\n'.format(test_loss, 100. * test_acc))
            
            self.history.append(epoch)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)
            self.test_loss.append(test_loss)
            self.test_acc.append(test_acc)
            
            self.save()
            
        print("Finish training for {} epochs".format(self.epochs))       