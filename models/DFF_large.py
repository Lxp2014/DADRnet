import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_networks import Encoder_MDCBlock1, Decoder_MDCBlock1

def make_model(args, parent=False):
    return Net()

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                    output_padding=output_padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class Net(nn.Module):
    def __init__(self, res_blocks=6, output_nc = 3):
        super(Net, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)

        self.dense0 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.fusion1 = Encoder_MDCBlock1(32, 2, mode='iter2')
        self.dense1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.fusion2 = Encoder_MDCBlock1(64, 3, mode='iter2')
        self.dense2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=1)

        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(128))

        self.convd8x = ConvLayer(128, 64, kernel_size=3, stride=1)

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2, padding=1,
                    output_padding=1)
        self.dense_2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.fusion_2 = Decoder_MDCBlock1(32, 2, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2, padding=1,
                    output_padding=1)
        self.dense_1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.fusion_1 = Decoder_MDCBlock1(16, 3, mode='iter2')

        self.conv_output = ConvLayer(16, output_nc, kernel_size=3, stride=1)


    def forward(self, input_x, flow="enc_dec"):
        if flow == "enc":
            x = input_x
            res1x = self.conv_input(x)
            feature_mem = [res1x]
            x = self.dense0(res1x) + res1x
            res2x = self.conv2x(x)
            res2x = self.fusion1(res2x, feature_mem)
            feature_mem.append(res2x)
            res2x =self.dense1(res2x) + res2x
            res4x =self.conv4x(res2x)
            res4x = self.fusion2(res4x, feature_mem)
            feature_mem.append(res4x)
            res4x = self.dense2(res4x) + res4x

            res4x_temp = self.conv8x(res4x)
            res4x = self.dehaze(res4x_temp)

            return res4x_temp, res4x

        elif flow == "dec":
            #  此时输入的应该和encoder编码维度一致的东西
            res4x = self.dehaze(input_x)
            res4x = self.convd8x(res4x)
            feature_mem_up = [res4x]

            res4x = self.convd4x(res4x)
            res2x = self.dense_2(res4x) + res4x
            res2x = self.fusion_2(res2x, feature_mem_up)
            feature_mem_up.append(res2x)
            res2x = self.convd2x(res2x)
            x = self.dense_1(res2x) + res2x
            x = self.fusion_1(x, feature_mem_up)
            x = self.conv_output(x)
            return x
