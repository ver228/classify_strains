#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:46:12 2017

@author: ajaver
"""
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class ResNetS(nn.Module):
    def __init__(self, 
                 block, 
                 layers, 
                 n_channels=2, 
                 num_classes=1000,
                 avg_pool_kernel = (7,2)
                 ):
        self.inplanes = 64
        super(ResNetS, self).__init__()
        
        self.conv1 = nn.Conv2d(n_channels, 
                               64, 
                               kernel_size=7, 
                               stride=2, 
                               padding=3,
                               bias=False
                               )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(avg_pool_kernel, stride=1)
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        #initialize all the modules, Xavier initialization for conv2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        #global max pooling
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    
    import os
    import sys
    import torch
    
    src_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'src')
    sys.path.append(src_dir)
    
    from classify.trainer import init_generator, Trainer, IS_CUDA
    
    if sys.platform == 'linux':
        log_dir_root = '/work/ajaver/classify_strains/results'
    else:        
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/'
    
    
    dataset = 'SWDB'
    
    params = dict(
            is_reduced = True,
            dataset = 'SWDB',
            data_file = None, #give the path of the .hdf5 location, otherwise it will use the defaults of my setup
            _valid_strains = ['JU258', 'CB4856'] #use for a quick test
    )
    
    gen_details, train_generator, test_generator = init_generator(**params)
    
    
    model = ResNetS(Bottleneck, 
                    [3, 4, 6, 3], 
                    n_channels=train_generator.n_channels, 
                    num_classes = train_generator.n_classes
                    )
    model_name = 'ResNet50'
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log_dir = os.path.join(log_dir_root, '{}_{}'.format(model_name, gen_details))
    if IS_CUDA:
        print('This is CUDA!!!!')
        torch.backends.cudnn.benchmark = True #useful for arrays of fix dimension
        model = model.cuda()
        criterion = criterion.cuda()
    
    n_epochs = 200
    t = Trainer(model,
             optimizer,
             criterion,
             train_generator,
             test_generator,
             n_epochs,
             log_dir
             )
    t.fit()
