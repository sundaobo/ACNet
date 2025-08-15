import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from vig_pytorch.gcn_lib.torch_vertex import Grapher_group
from loss.losses import Mutual_info_reg
import numpy as np

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        #self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        #x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class WF(nn.Module):
    def __init__(self, in_ch, out_ch, num_heads=8, window_size=8):
        super(WF, self).__init__()
        self.wt = DWTForward(J=1, mode='reflect', wave='haar')
        self.it = DWTInverse(mode='reflect', wave='haar')

        self.CA_L = Grapher_group(in_ch)
        self.CA_H = Grapher_group(3*in_ch)


        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(3*in_ch, 3*out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(3*out_ch),
            nn.ReLU(inplace=True),
        )

        self.mlpX = MLP(dim=in_ch, mlp_ratio=4)

        self.mlpwW = MLP(dim=3*in_ch, mlp_ratio=4)

        self.mlpwL = MLP(dim=in_ch, mlp_ratio=4)

        self.mlpw3 = MLP(dim=3*in_ch, mlp_ratio=4)

        self.conv1 = ConvBnLeakyRelu2d(3*in_ch, in_ch)
        self.conv2 = ConvBnLeakyRelu2d(3*in_ch, 4*in_ch)

        self.convdwL = nn.Conv2d(in_ch,out_ch, 7, padding=3, groups=out_ch)  # dw
        self.actL = nn.GELU()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.convHFeature = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.manloss4 = Mutual_info_reg(in_ch, in_ch)
    
    def forward(self, x,imagename=None):

        # yL, (y_HL, y_LH, y_HH) = self.wt(x)
        yL, yH = self.wt(x)
        yy = yL
        hh = yH
       
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = self.mlpX(x)

        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        loss = self.manloss4(yL, yH)
        yH = self.CA_H(yH)
        yH = self.mlpwW(yH)
        yH = self.conv1(yH)

        yL = self.CA_L(yL)
        yL = self.mlpwL(yL)

        diff = torch.abs(yH - yL)
        y = torch.cat([yL, yH, diff], dim=1)
        y = self.mlpw3(y)
        y = self.conv2(y)

        # ya, yh, yv, yd = torch.chunk(y, 4, dim=1)
        # y = self.it([ya, (yh, yv, yd)], None)


        ya, yh, yv, yd = torch.chunk(y, 4, dim=1)
        reconstructed_yh = torch.stack([yh, yv, yd], dim=2)  # 在 dim=2 堆叠
        coeffs = (ya, [reconstructed_yh])
        y = self.it(coeffs)
        y = y * self.sca(y)
        y = y + x
        y = self.convHFeature(y)
        # loss = self.get_wavelet_loss()

        return y ,loss


