from torch.utils.data import dataset
from PIL import Image
from torchvision import transforms, models
import random
import numpy as np
import torch

size = 512
trans = {
        'train':
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.ColorJitter(brightness=0.126, saturation=0.5),
                # transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), fillcolor=0, scale=(0.8, 1.2), shear=None),
                transforms.Resize((int(size / 0.875), int(size / 0.875))),
                transforms.RandomCrop((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ]),
        'val':
            transforms.Compose([
                transforms.Resize((int(size / 0.875), int(size / 0.875))),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }


class Dataset(dataset.Dataset):
    def __init__(self, mode):
        assert mode in ['train', 'val']
        txt = './%s.txt' % mode

        fpath = []
        labels = []
        with open(txt, 'r')as f:
            for i in f.readlines():
                fp, label = i.strip().split(',')
                fpath.append(fp)
                labels.append(int(label))

        self.fpath = fpath
        self.labels = labels
        self.mode = mode
        self.trans = trans[mode]
        
    def __getitem__(self, index):
        fp = self.fpath[index]
        label = self.labels[index]
        img = Image.open(fp).convert('RGB')
        if self.trans is not None:
            img = self.trans(img)

        return img, label

    def __len__(self):
        return len(self.labels)
    