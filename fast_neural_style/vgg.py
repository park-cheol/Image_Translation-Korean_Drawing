from collections import namedtuple
'''tuple vs namedtuple
tuple: 튜플에 있는 요소를 접근할 때 순서를 기억해서 접근해야하는 단점있음
namedtuple: 키와 인덱스로 접근가능/ dic과 비슷/ 메모리 효율
ex) a = namedtuple('d', ['x','y'])
p = a(11, y=22) 키워드나 위치로 접근가능
print ==> p = d(x=11, y=22)
'''
import torch
import torch.nn as nn
from torchvision import models

#TODO VGG16을 나중에 다른 CNN Network로도 해볼 것(VGG19, RESNET .etc)

class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        # Maxpool 기준으로 layer 나눔
        for x in range(4): # layer 1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
            # Sequential을 [x]로 호출가능
        for x in range(4, 9): # layer 2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16): # layer3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23): # layer4
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad: # 즉 True 경우
            for param in self.parameters():
                param.requires_grad = False
                # True였던 것을 다시 False로 바꿔줌(가중치를 Fix)

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h # 1 layer 2번째 Conv
        h = self.slice2(h)
        h_relu2_2 = h # 2 layer 2번째 Conv
        h = self.slice3(h)
        h_relu3_3 = h # 3 layer 3번째 Conv
        h = self.slice4(h)
        h_relu4_3 = h # 4 layer 3번째 Conv

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out










