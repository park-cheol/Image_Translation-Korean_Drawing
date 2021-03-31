import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

import PIL
import matplotlib.pyplot as plt

# 설정한 content layer까지의 결과를 이용해 content loss를 계산
if name in content_layers:
    target_feature = model(content_img).detach()
    # 현재 model까지만 실행
    content_loss = ContentLoss(target_feature)
# 그 loss를 구함
    model.add_module("content_loss_{}".format(i), content_loss)
    content_losses.append(content_loss)

# 설정한 style layer까지의 결과를 이용해 style loss를 계산
if name in style_layers:
    target_feature = model(style_img).detach()
    style_loss = StyleLoss(target_feature)
    model.add_module("style_loss_{}".format(i), style_loss)
    style_losses.append(style_loss)
if name in content_layers:
    target_feature = model(content_img).detach()
    # 현재 model까지만 실행
    content_loss = ContentLoss(target_feature)
    # 그 loss를 구함
    model.add_module("content_loss_{}".format(i), content_loss)
    content_losses.append(content_loss)

    # 설정한 style layer까지의 결과를 이용해 style loss를 계산
if name in style_layers:
    target_feature = model(style_img).detach()
    style_loss = StyleLoss(target_feature)
    model.add_module("style_loss_{}".format(i), style_loss)
    style_losses.append(style_loss)











