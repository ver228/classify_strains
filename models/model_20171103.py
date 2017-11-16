# -*- coding: utf-8 -*-
"""
Code modified from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

"""

import torch.nn as nn
import torch.nn.functional as F
import math
import tqdm


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
    import torch
    import sys
    
    
    # add the 'src' directory as one where we can import modules
    src_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'classify_nn')
    sys.path.append(src_dir)
    
    from data import SkeletonsFlowFull, get_valid_strains
    
    models_path = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/model_20171103/'
    models_names = [
            'resnet50_R_CeNDR_ang__S10_F0.04_20171104_182812.pth.tar',
            'resnet50_R_CeNDR_ang__S20_F0.1_20171102_210004.pth.tar',
            'resnet50_R_CeNDR_ang__S90_F0.04_20171104_182641.pth.tar',
            'resnet50_R_CeNDR_ang__S90_F0.1_20171102_210003.pth.tar'
            ]
    
    
    n_classes = 197
    
    model_name = models_names[2]
    fname = os.path.join(models_path, model_name)
    
    model = ResNetS(Bottleneck, [3, 4, 6, 3], n_channels=1, num_classes = n_classes)
    checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    dataset = 'CeNDR'
    data_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/_old/CeNDR_skel_smoothed.hdf5'
    #data_file = _h_get_datset_file(dataset)
    valid_strains = get_valid_strains(dataset, is_reduced=True)
    
    dd = model_name.split('_')
    sample_size_seconds = [float(x[1:]) for x in dd if x.startswith('S')][0]
    sample_frequency_s = [float(x[1:]) for x in dd if x.startswith('F')][0]
    
    gen = SkeletonsFlowFull(
                          n_batch = 32, 
                          data_file = data_file,
                          set_type = 'test', 
                          sample_size_seconds = sample_size_seconds, 
                          sample_frequency_s = sample_frequency_s,
                          valid_strains = valid_strains,
                          is_torch = True
                          )
    results = []
    for input_v, target in tqdm.tqdm(gen):
        output = model.forward(input_v)
        _, pred1 = torch.max(output, 1)
        
        results.append((target.data.numpy(), pred1.data.numpy()))
        
        
    #%%
    #NOTE it seems that I train the models using shifting the strain_id by one...
    #This shouldn't affect the embeddings, I'll correct it next time I train a model...
    
    import numpy as np
    y_true, y_pred = map(np.concatenate, zip(*results))
    #chunk accuracy
    print(np.sum(y_true==y_pred-1)/y_true.size)
    
    #%%
    '''
    0.691 -> resnet50_R_CeNDR_ang__S10_F0.04_20171104_182812.pth.tar
    0.715 -> resnet50_R_CeNDR_ang__S20_F0.1_20171102_210004.pth.tar
    0.801 -> resnet50_R_CeNDR_ang__S90_F0.04_20171104_182641.pth.tar
    0.791 -> resnet50_R_CeNDR_ang__S90_F0.1_20171102_210003.pth.tar
    '''