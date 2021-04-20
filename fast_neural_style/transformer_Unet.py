import torch
import torch.nn as nn
import torch.nn.functional

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.down1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.down2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.down3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.LeakyReLU(0.25)

    def forward(self, X):
        d1 = self.down1(X)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        y = self.res1(d3)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class ConvLayer(nn.Module): # 크기가 1/2 downsampling
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2 # padding 후 같은 차원
        layers = [
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        out = self.model(x)
        return out

#TODO Bottleneck 구조로
class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out = out + residual

        return out

class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
        Upsamples the input and then does a convolution. This method gives better results
        compared to ConvTranspose2d.
        ref: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, normalize=True):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        layers = [
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x_in = x
        if self.upsample: # 아마도 Super resolution 할려고 하는 것 같다
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        # functional.interpolate(input, mode='', sclae_factor)
        # Down or up sampling
        out = self.model(x_in)
        out = torch.cat((out, skip_input), 1)
        return out











