import argparse
import os
import sys
import time
import re
"""re(regular expression): 정규 표현식을 지원하기 위해
re.match('b','bba'): match되어 object반환 / ('b',aab') : None
re.search('b','aab'): 도 match됌 차이 파악
span=(0,1) 인덱스 0번부터 1번문자 전까지 매치돼었음
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx # caffe 사용시 사용 우리는 pytorch사용

import utils
from transformer_net import TransformerNet
from vgg import Vgg16
import cv2 as cv

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir): # 존재하지 않을시 생성
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            """checkpoint_model_dir 인자를 받았지만 경로에 파일이 존재하지안을 시"""
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)
    """except 오류를 모르시 그냥 print(.....)
    예상이 가능한 오류항목 추가하여 처리 except FileNotFoundError: 파일찾을수 없다
    OSError : OS 에러
    except 말고 finally도 있음 ==> 오류가 발생해서 프로그램이 종료되더라고 꼭 실행하라
    """

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)) # 255씩 곱해준다?
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    #def load_image(filename, size=None, scale=None):
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style) # 전처리
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)
    # batch 목적 차원 하나 늘리고, 1,1,1은 각 그 차원에 곱을 해줌
    # [3, 512 ,512] -> [128, 2, 1, 1] -> [128, 6, 512, 512]

    # style(img) 정규화하고 vgg
    features_style = vgg(utils.normalize_batch(style))
    # return: vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3) features map
    gram_style = [utils.gram_matrix(y) for y in features_style]
    # 각 리스트에서 하나씩 뽑아 gram_matrix하여 리스트안에 넣고

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss =0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            # batch_id = iteration  x[batch, channels ,size, size]
            n_batch = len(x) # tensor[128 ,3, 64 ,64] 경우 ==> len(x)=128 배치크기나타냄
            count += n_batch # 1루프마다 배치 더하니까 iteration
            optimizer.zero_grad()

            # COCO dataset
            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)


            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            # gram_style = [utils.gram_matrix(y) for y in features_style] 스타일이미지
            # features_y = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3) coco
            for ft_y, gm_s in zip(features_y, gram_style): # 각 list에 한개씩 뽑아봄
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval ==0:
                message = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1))
                print(message)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu() # 저장할때 eval과 cpu로
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                # 디렉토리경로와 파일이름
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

def stylize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    # batch 목적으로 한차원 추가 = 차원 같게 해주기위해서

    with torch.no_grad():
        style_model = TransformerNet() # 생성
        state_dict = torch.load(args.model) # 불러오고
        # torch.load 경로이고 torch.load_state_dict(객체)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k] # 그 파트 삭제
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        output = style_model(content_image).cpu()
    # 저장은 CPU로
    utils.save_image(args.output_image, output[0])







def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for F-N-S")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, required=True, help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 512 X 512")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)



if __name__ == "__main__":
    main()