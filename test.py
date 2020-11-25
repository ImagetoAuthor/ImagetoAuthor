#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import time, datetime
import pdb, traceback

# import imagehash
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from resnest.torch import resnest101

# In[2]:


import torch
# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter


class MyDataset(Dataset):
    def __init__(self, test_jpg):
        self.test_jpg = test_jpg

        self.transforms = transforms.Compose([
            transforms.Resize(660),
            transforms.RandomCrop(600),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img = Image.open(self.test_jpg[index]).convert('RGB')
        img = self.transforms(img)

        return img, torch.from_numpy(np.array(int('H' in self.test_jpg[index])))

    def __len__(self):
        return len(self.test_jpg)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.base = EfficientNet.from_pretrained('efficientnet-b3')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._max_pooling = nn.AdaptiveMaxPool2d(1)
        num_ftrs = self.base._fc.in_features
        self.reduce_layer = nn.Conv2d(num_ftrs * 2, 512, 1)  # b3 num_ftrs=1536
        self._dropout = nn.Dropout(0.3)
        self._fc = nn.Linear(512, 49)

    def forward(self, x):
        x = self.base.extract_features(x)
        x1 = self._avg_pooling(x)
        x2 = self._max_pooling(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.reduce_layer(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x


# In[5]:


def predict(test_loader, model, tta=9):
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(test_loader):
                inputs = inputs.cuda()
                target = target.cuda()

                output = model(inputs)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
        # print(test_pred_tta)

    return test_pred_tta


# In[11]:


test_jpg = ['test/{0}.jpg'.format(x) for x in range(800)]
test_jpg = np.array(test_jpg)

test_data = MyDataset(test_jpg)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=10, pin_memory=True)

# model = Net().cuda()
model = EfficientNet.from_pretrained('efficientnet-b3')
num_trs = model._fc.in_features
model._fc = nn.Linear(num_trs, 49)
model = model.cuda()

test_pred = 0.

pth = 'pth/resnet.pth'
model.load_state_dict(torch.load(pth))
test_pred = predict(test_loader, model, tta=9)

# In[ ]:


test_csv = pd.DataFrame()
test_csv['num'] = list(range(800))
test_csv['label'] = np.argmax(test_pred, 1)

test_csv.to_csv('result.csv', index=None, header=False)
print('successfule load!')