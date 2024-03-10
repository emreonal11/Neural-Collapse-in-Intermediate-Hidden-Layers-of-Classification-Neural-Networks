import gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import torch
from tqdm import tqdm
from collections import OrderedDict
import pickle

official_names = {}

official_names['datasets'] = [
    'MNIST',
    'FashionMNIST',
    'CIFAR10',
    'CIFAR100',
    'TinyImageNet',
    'SVHN'
]

official_names['models'] = [
    'VGG8',
    'VGG11',
    'VGG13_scum',
    'VGG13',
    'Densenet121',
    'MLP6',
    'Resnet18'
]

official_names['losses'] = [
    'MSE',
    'BCE'
]

official_names['activations'] = [
    'ReLU',
    'LeakyReLU',
    'Tanh',
    'Sigmoid',
    'SiLU',
]


def convert_activations(model, old_activation, new_activation):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_activations(module, old_activation, new_activation)
        if type(module) == old_activation:
            model._modules[name] = new_activation
    return model


def get_model(model_name, num_classes = 10, activation='ReLU'):
    if model_name not in official_names['models']:
        raise Exception(f'Model {model_name} not in model list.')
    if model_name == 'VGG13_scum':
        model = VGG13_scum(num_classes)
    elif model_name == 'VGG13':
        model = VGG13(num_classes)
    elif model_name == 'VGG8':
        model = VGG8(num_classes)
    elif model_name == 'VGG11':
        model = VGG11(num_classes)
    elif model_name == 'MLP6':
        model = MLP6(num_classes)
    elif model_name == 'Resnet18':
        model = models.resnet18()
        model.conv1 = nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'Densenet121':
        model = models.densenet121()
        model.features.conv0 = nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(1024, num_classes)
    else:
        raise Exception(f'Model {model_name} get_model not implemented.')
    
    if activation != 'ReLU':
        if activation == 'Tanh':
            model = convert_activations(model, nn.ReLU, nn.Tanh())
        elif activation == 'LeakyReLU':
            model = convert_activations(model, nn.ReLU, nn.LeakyReLU())
        elif activation == 'Sigmoid':
            model = convert_activations(model, nn.ReLU, nn.Sigmoid())
        elif activation == 'SiLU':
            model = convert_activations(model, nn.ReLU, nn.SiLU())
    return model


#######################################################################
#######################################################################
# MLP6 #
#######################################################################
#######################################################################

class MLP6(nn.Module):
    def __init__(self, C):
        super(MLP6, self).__init__()
        self.fc1 = nn.Sequential(nn.LazyLinear(4096) , nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096,4096), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(4096,4096), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(4096,4096), nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(4096,4096), nn.ReLU())
        self.fc6 = nn.Linear(4096, C)

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        output = self.fc6(x)
        return output
    

#######################################################################
#######################################################################
# VGG8 #
#######################################################################
#######################################################################

class VGG8(nn.Module):
    def __init__(self, C):
        super(VGG8, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.LazyConv2d(
                #in_channels=3,              
                out_channels=64,            
                kernel_size=3,      #3        
                stride=1,           #1         
                padding=1,          #1      
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(64, 128, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(128, 256, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),   
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.fc1 = nn.Sequential(nn.LazyLinear(4096) , nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096,4096), nn.ReLU())
        self.fc3 = nn.Linear(4096,C)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output



#######################################################################
#######################################################################
# VGG11 #
#######################################################################
#######################################################################

class VGG11(nn.Module):
    def __init__(self, C):
        super(VGG11, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # fully connected layer, output 10 classes
        self.fc1 = nn.Sequential(nn.LazyLinear(4096) , nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096,4096), nn.ReLU())
        self.fc3 = nn.Linear(4096,C)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output

#######################################################################
#######################################################################
# VGG13_scum #
#######################################################################
#######################################################################

class VGG13_scum(nn.Module):
    def __init__(self, C):
        super(VGG13_scum, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Adding 2 extra conv blocks
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
        )
        
        # fully connected layer, output 10 classes
        self.fc1 = nn.Sequential(nn.LazyLinear(4096) , nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096,4096), nn.ReLU())
        self.fc3 = nn.Linear(4096,C)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        
        x = x.view(x.size(0), -1)  

        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output
    
    
#######################################################################
#######################################################################
# VGG13 # VGG 13 with potentially one extra pooling layer in conv1 block
#######################################################################
#######################################################################

class VGG13(nn.Module):
    def __init__(self, C):
        super(VGG13, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # fully connected layer, output 10 classes
        self.fc1 = nn.Sequential(nn.LazyLinear(4096) , nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096,4096), nn.ReLU())
        self.fc3 = nn.Linear(4096,C)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        
        x = x.view(x.size(0), -1)  

        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output
    


