import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np
from efficientnet_pytorch import EfficientNet
import random


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=(1,1)):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
        
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class GeneralizedMeanPooling(nn.Module):
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super().__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'



class BaseModel(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True, pool_type='max', down=True, metric='linear'):
        super().__init__()
        self.model_name = model_name
        
        if model_name == 'eff-b3':
            backbone = EfficientNet.from_pretrained('efficientnet-b3')
            plane = 1536
        elif model_name == 'resnext50':
            backbone = nn.Sequential(*list(models.resnext50_32x4d(pretrained=pretrained).children())[:-2])
            plane = 2048
        else:
            backbone = None
            plane = None

        self.backbone = backbone
        
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == 'cat':
            self.pool = AdaptiveConcatPool2d()
            down = 1
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pool_type == 'gem':
            self.pool = GeneralizedMeanPooling()
        else:
            self.pool = None
        
        if down:
            if pool_type == 'cat':
                self.down = nn.Sequential(
                    nn.Linear(plane * 2, plane),
                    nn.BatchNorm1d(plane),
                    nn.Dropout(0.2),
                    nn.ReLU(True)
                    )
            else:
                self.down = nn.Sequential(
                    nn.Linear(plane, plane),
                    nn.BatchNorm1d(plane),
                    nn.Dropout(0.2),
                    nn.ReLU(True)
                )
        else:
            self.down = nn.Identity()
        
        self.se = SELayer(plane)
        self.hidden = nn.Linear(plane, plane)
        self.relu = nn.ReLU(True)
        
        if metric == 'linear':
            self.metric = nn.Linear(plane, num_classes)
        elif metric == 'am':
            self.metric = AddMarginProduct(plane, num_classes)
        else:
            self.metric = None

    def forward(self, x):
        if self.model_name == 'eff-b3':
            feat = self.backbone.extract_features(x)
        else:
            feat = self.backbone(x)
        
        feat = self.pool(feat)
        se = self.se(feat).view(feat.size(0), -1)
        feat_flat = feat.view(feat.size(0), -1)
        feat_flat = self.relu(self.hidden(feat_flat) * se)

        out = self.metric(feat_flat)
        return out


if __name__ == '__main__':
    model = BaseModel(model_name='eff-b3').eval()
    x = torch.randn((1, 3, 224, 224))
    out = model(x)
    print(out.size())
    print(model)