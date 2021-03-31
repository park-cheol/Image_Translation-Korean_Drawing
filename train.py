import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt

transforms_ = [
    transforms.Resize(256, Image.BICUBIC), #https://m.blog.naver.com/PostView.nhn?blogId=dic1224&logNo=220840978075&proxyReferer=https:%2F%2Fwww.google.com%2F

    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]
dataset = datasets.ImageFolder("data/train", transform=transforms.Compose(transforms_))
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

data = iter(dataloader)
images, labels = data.next()

def imshow(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
imshow(torchvision.utils.make_grid(images, nrow=5))


transforms__ = [
    transforms.Resize(512, Image.BICUBIC), #https://m.blog.naver.com/PostView.nhn?blogId=dic1224&logNo=220840978075&proxyReferer=https:%2F%2Fwww.google.com%2F

    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]
datasets = datasets.ImageFolder("data/train", transform=transforms.Compose(transforms__))
dataloaders = DataLoader(datasets, batch_size=10, shuffle=False)

data = iter(dataloaders)
images1, labels1 = data.next()

def imshow(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
imshow(torchvision.utils.make_grid(images1, nrow=5))







