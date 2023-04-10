from __future__ import absolute_import
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

# from IPython import embed

# class ResNet50(nn.Module):
#     def __init__(self, num_classes, loss = {'softmax, metric'}, **kwargs):
#         super(ResNet50, self).__init__()
#         resnet50 = torchvision.models.resnet50(pretrained = True)
#         self.base = nn.Sequential(*list(resnet50.children())[:-2])   #去掉最后两层,定义自己的分类器
#         self.classifier = nn.Linear(2048, num_classes)
        
#     def forward(self, x):
#         x = self.base(x)
        
#         return x

import torch.nn as nn

class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2d, self).__init__()


    def forward(self, x):
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x,kernel_size= (1, inp_size[3]))


class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        #f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        if not self.training:
            return f,lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return  f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
            
    

if __name__ == '__main__':
    model = ResNet50(num_classes=751)
    imgs = torch.Tensor(32, 3, 256 , 128)
    f = model(imgs)