import torch
from PIL import Image

"""Interpolation(보간법): 실제 값과 실제값 사이의 결과예측값을 부드럽게 이어주는 방법
ANTIALIAS(안티 에일리어싱): 이미지는 작은 사각형 모양의 점인 픽셀로 이루어져있음 이로인해 곡선같은
선을 확대해보면 사각형모양들이 모서리모양으로 맞대고 이어져 빈공간 발생 이러한 빠진 데이터를
보충하고 주변에 중간 색의 픽셀을 추가해 자연스럽게 이미지 표헌 기술

BICUBIC: spline 보간법 사용-> 구간을 작게 나누어 저차원 다항식으로 함수 구하는 방법
                            고차원일 시 가운데는 잘 표현되지만 바깥쪽이 많이 요동치므로
            
"""


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy() # 픽셀 값범위 맞춰주기 위해
    # Tensor [C, H, W] img [H, W, C]
    img = img.transpose(1, 2, 0).astype("uint8")
    # astpye로 dtype 변환 가능 uint8은 부호가없는 8비트 정수 (<=> int8 부호가있는 8비트 정수)
    img = Image.fromarray(img)
    # numpy 배열을 PIL이미지로 변환
    img.save(filename)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h) # [b, ch, M]
    features_t = features.transpose(1, 2) # [b, M, ch]
    gram = features.bmm(features_t) / (ch * h * w)
    '''bmm: batch matrix-matrix product of matrices
    input과 mat는 반드시 3-DTensor 같은 matrices 가지고있는
    (bxnxm),(bxmxp) ==> (bxnxp)'''
    return gram

def normalize_batch(batch):
    """ normalize using imagenet mean and std"""
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) # batch와 같은차원을 위해
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0) # tensor의 범위(0~1) 밤추기위해
    # div의 inplace version
    return (batch - mean) / std