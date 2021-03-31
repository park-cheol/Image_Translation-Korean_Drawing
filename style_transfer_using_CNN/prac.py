import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

import PIL
import matplotlib.pyplot as plt

import copy
from torchvision.utils import save_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(img_path, imsize):
    loader = transforms.Compose([
        transforms.Resize(imsize), # 이미지 크기 설정
        transforms.ToTensor()
        # ToTensor: PIL image or numpy(H,W,C) 범위[0, 255] -> [0.0, 1.0] 변경
    ])
    image = PIL.Image.open(img_path)
    image = loader(image).unsqueeze(0) # 배치 차원 추가
    return image.to(device, torch.float) # Gpu 가동

# torch.tensor형태의 이미지를 화면에 출력
def imshow(tensor):
    image = tensor.cpu().clone() # matplotlib은 cpu기반
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.show()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

content_layers = ['conv_1']

def content_reconstruction(cnn, content_img, input_img, iters):
    model, content_losses = get_content_losses(cnn, content_img, input_img)
    optimizer = optim.Adam([input_img.requires_grad_()])

    print("[ Start ]")

    # 하나의 값만 이용하기 위해 배열 형태로 사용
    run = [0]
    while run[0] <= iters:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            content_score = 0

            for cl in content_losses:
                content_score += cl.loss

            content_score.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"[ Step: {run[0]} / Content loss: {content_score.item()}]")

            if run[0] % 5000 == 0:
                save_image(input_img.cpu().detach()[0], 'content_test_%s.png'% (run[0]))
            return content_score

        optimizer.step(closure)

    # 결과적으로 이미지의 각 픽셀의 값이 [0, 1] 사이의 값이 되도록 자르기
    input_img.data.clamp_(0, 1)

    return input_img

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def get_content_losses(cnn, content_img, noise_image):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
    content_losses = []

    # 가장 먼저 입력 이미지가 입력 정규화(input normalization)를 수행하도록
    model = nn.Sequential()

    # 현재 CNN 모델에 포함되어 있는 모든 레이어를 확인하며
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            name = 'conv_{}'.format(i)
            i += 1
            model.add_module(name=name, module=layer)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
            model.add_module(name=name, module=layer)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            model.add_module(name=name, module=layer)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
            model.add_module(name=name, module=layer)
        else:
            for bottleneck in layer.children():
                for instance in bottleneck.children():
                    if isinstance(instance, nn.Conv2d):
                        name = 'conv_{}'.format(i)
                        i += 1
                    elif isinstance(instance, nn.ReLU):
                        name = 'relu_{}'.format(i)
                        instance = nn.ReLU(inplace=False)
                    elif isinstance(instance, nn.MaxPool2d):
                        name = 'pool_{}'.format(i)
                    elif isinstance(instance, nn.BatchNorm2d):
                        name = 'bn_{}'.format(i)
                    elif isinstance(instance, nn.Sequential):
                        continue
                    else:
                        raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
                    model.add_module(name=name, module=instance)

                    # 설정한 content layer까지의 결과를 이용해 content loss를 계산
                    if name in content_layers:
                        target_feature = model(content_img).detach()
                        # 현재 model까지만 실행
                        content_loss = ContentLoss(target_feature)
                        # 그 loss를 구함
                        model.add_module("content_loss_{}".format(i), content_loss)
                        content_losses.append(content_loss)
    print(model)
    # 마지막 loss 이후의 레이어는 사용하지 않도록
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss):
            break
    model = model[:(i + 1)]
    return model, content_losses

cnn = models.resnet50(pretrained=True).to(device).eval()
content_img = image_loader('image/content_img_2.jpg', (512, 512))
input_img = torch.empty_like(content_img).uniform_(0, 1).to(device)

output = content_reconstruction(cnn, content_img=content_img, input_img=input_img, iters=30000)
imshow(output)






