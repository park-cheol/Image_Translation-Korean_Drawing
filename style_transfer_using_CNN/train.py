import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

import PIL
import matplotlib.pyplot as plt

import copy
'''
a = b --> 둘 다 동일한 메모리 가리킴 = a가 바뀔시 b도바뀜
b = a.copy.copy(a) --> 다른 메모리 가지게 copy
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지를 불러 다운받아 텐서로 변환
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

'''
# target image 불러오기
img_path = 'image/content_img1.jpeg'
target_image = image_loader(img_path, (512, 512))

# 노이즈 이미지 준비
# target_image shape와 같은 텐서 생성 , 균등분포 0~1사이, gradient 추적 X
noise = torch.empty_like(target_image).uniform_(0, 1).to(device)

loss = nn.MSELoss()
iter = 10
lr = 1e4

print("[start]")

for i in range(iter):
    noise.requires_grad = True

    output = loss(noise, target_image)
    output.backward()

    # 계산된 기울기로 loss가 감소하는 방향으로 업데이트
    gradient = lr * noise.grad
    noise = torch.clamp(noise - gradient, min=0, max=1).detach_()
    # detach_() == detach()은 inplace version
    # inplace operation = copy하여 만드는 것이 아닌 주어진 tensor에 직접적으로 변화를 주는 연산
    # inplace는 항상 _ 라는 접두사 사용 ( += , *= 도 inplace 방식)
    # torch.clamp(input, min, max) : input이 [min, max]사이로 오게함
    # 즉 노이즈의 각필셀값이 0~1 사이오게 자름

    if (i + 1) % 10 ==0:
        print(f'[step: {i+1} ]')
        print(f'[loss: {output}')
'''

'''
# 스타일 입히기
content_img = image_loader('image/content_img_2.jpg', (512, 512))
print(content_img.size())
style_img = image_loader('image/style_img_2.png', (512, 512))
print(style_img.size())

cnn = models.vgg19(pretrained=True).features.to(device).eval()
# eval 모드 output features값만 가지고 오도록함
'''
# 입력 정규화(Normalization)를 위한 초기화
# 입력으로 들어온 이미지를 정규화 한 뒤 결과를 구할 수 있는 형태
# 마찬가지로 테스트 할때도 정규화 할 필요가 있음
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std



def gram_matrix(input):
    # a는 배치 크기, b는 특징 맵의 개수, (c, d)는 특징 맵의 차원을 의미
    a, b, c, d = input.size()
    # 논문에서는 i = 특징 맵의 개수, j = 각 위치(position)
    features = input.view(a * b, c * d)
    # 행렬 곱으로 한 번에 Gram 내적 계산 가능
    G = torch.mm(features, features.t())
    # Normalize 목적으로 값 나누기
    return G.div(a * b * c * d)


# 스타일 손실(style loss) 계산을 위한 클래스 정의
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        # target의 gram_matrix 구하고

    def forward(self, input): # 차후 input의 gram_matrix와 loss
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)

        return input

'''
style_layers = ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']
'''
'''
# 스타일 손실(style loss)을 계산하는 함수
def get_style_losses(cnn, style_img_, noise_image):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
    # 정규화
    style_losses = []

    # 가장 먼저 입력 이미지가 입력 정규화(input normalization)를 수행하도록
    model = nn.Sequential(normalization)

    # 현재 CNN 모델에 포함되어 있는 모든 레이어를 확인하며 이름 지어줌
    i = 0
    for layer in cnn.children(): # 각 feature conv2d나 relu 같은것 리스트로
        if isinstance(layer, nn.Conv2d):
            i += 1 # i를 1씩 증가
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # 설정한 style layer까지의 결과를 이용해 style loss를 계산
        if name in style_layers: # 위에서 정한 style_layers 리스트중에 name이 같으면
            target_feature = model(style_img_).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss) # module에 loss텀 추가
            style_losses.append(style_loss)
            # 리스트에도 추가

    # 마지막 style loss 이후의 레이어는 사용하지 않도록
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)] # model에서 쓸부분만 loss 앞가져옴
    return model, style_losses
'''
'''
def style_reconstruction(cnn, style_img, input_img, iters):
    model, style_losses = get_style_losses(cnn, style_img, input_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()]) # adam을 더많이씀

    print("[ Start ]")

    # 하나의 값만 이용하기 위해 배열 형태로 사용
    run = [0]
    while run[0] <= iters:

        def closure():
            input_img.data.clamp_(0, 1) # 범위를 넘어가지않도록 잘라줌

            optimizer.zero_grad()
            model(input_img)
            style_score = 0

            for sl in style_losses:
                style_score += sl.loss

            style_score *= 1e5
            style_score.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"[ Step: {run[0]} / Style loss: {style_score.item()}]")

            return style_score

        optimizer.step(closure)

    # 결과적으로 이미지의 각 픽셀의 값이 [0, 1] 사이의 값이 되도록 자르기
    input_img.data.clamp_(0, 1)

    return input_img


# 콘텐츠 이미지와 동일한 크기의 노이즈 이미지 준비하기
input_img = torch.empty_like(content_img).uniform_(0, 1).to(device)

# style reconstruction 수행
print(style_img.size(), input_img.size())
output = style_reconstruction(cnn, style_img=style_img, input_img=input_img, iters=10)
'''


# 콘텐츠 손실(content loss) 계산을 위한 클래스 정의
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

'''
content_layers = ['conv_4']


# 콘텐츠 손실(content loss)을 계산하는 함수
def get_content_losses(cnn, content_img, noise_image):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
    content_losses = []

    # 가장 먼저 입력 이미지가 입력 정규화(input normalization)를 수행하도록
    model = nn.Sequential(normalization)

    # 현재 CNN 모델에 포함되어 있는 모든 레이어를 확인하며
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # 설정한 content layer까지의 결과를 이용해 content loss를 계산
        if name in content_layers:
            target_feature = model(content_img).detach()
            content_loss = ContentLoss(target_feature)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

    # 마지막 content loss 이후의 레이어는 사용하지 않도록
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss):
            break

    model = model[:(i + 1)]
    return model, content_losses
'''
'''
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

            return content_score

        optimizer.step(closure)

    # 결과적으로 이미지의 각 픽셀의 값이 [0, 1] 사이의 값이 되도록 자르기
    input_img.data.clamp_(0, 1)

    return input_img


# 콘텐츠 이미지와 동일한 크기의 노이즈 이미지 준비하기
input_img = torch.empty_like(content_img).uniform_(0, 1).to(device)



# content reconstruction 수행
output = content_reconstruction(cnn, content_img=content_img, input_img=input_img, iters=10)
'''
content_layers = ['conv_1']
style_layers = ['conv_1', 'conv_10', 'conv_20', 'conv_30']
# 0 12 34 56

# Style Transfer 손실(loss)을 계산하는 함수
def get_losses(cnn, content_img, style_img, noise_image):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
    content_losses = []
    style_losses = []

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

                    # 설정한 style layer까지의 결과를 이용해 style loss를 계산
                    if name in style_layers:
                        target_feature = model(style_img).detach()
                        style_loss = StyleLoss(target_feature)
                        model.add_module("style_loss_{}".format(i), style_loss)
                        style_losses.append(style_loss)

    print(model)
    # 마지막 loss 이후의 레이어는 사용하지 않도록
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, content_losses, style_losses

from torchvision.utils import save_image

def style_transfer(cnn, content_img, style_img, input_img, iters):
    model, content_losses, style_losses = get_losses(cnn, content_img, style_img, input_img)
    optimizer = optim.Adam([input_img.requires_grad_()])
    # requires_grad_() 호출 시 requires_grad가 True로 설정

    print("[ Start ]")

    # 하나의 값만 이용하기 위해 배열 형태로 사용
    run = [0]
    while run[0] <= iters:
        # closure : contentloss와 styleloss를 같이 계산하고 더해주기 위한 loss
        # optimizer.step(closure) closure : gradient
        def closure():
            if run[0] % 10 == 0:
                save_image(input_img.cpu().detach()[0], 'image/output5_%s.png' % (run[0]))
                print("이미지저장")
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            content_score = 0
            style_score = 0
            for cl in content_losses:
                content_score += cl.loss
            for sl in style_losses:
                style_score += sl.loss

            style_score *= 1e5
            loss = content_score + style_score
            loss.backward()

            run[0] += 1
            if run[0] % 100 == 0:
                print(f"[ Step: {run[0]} / Content loss: {content_score.item()} / Style loss: {style_score.item()}]")


            return content_score + style_score

        optimizer.step(closure)

    # 결과적으로 이미지의 각 픽셀의 값이 [0, 1] 사이의 값이 되도록 자르기
    input_img.data.clamp_(0, 1)

    return input_img

content_img = image_loader('image/content_img_2.jpg', (512, 512))
print(content_img.size())
style_img = image_loader('image/style_img_2.png', (512, 512))
print(style_img.size())
cnn = models.resnet34(pretrained=True).to(device).eval()

# noise 이미지 생성
input_img = torch.empty_like(content_img).uniform_(0, 1).to(device)

output = style_transfer(cnn, content_img=content_img, style_img=style_img, input_img=input_img, iters=100000)
imshow(output)









