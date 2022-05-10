import torch 
from torch import nn
#import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = False):
        super(DoubleConv, self).__init__()
        #self.out_channels = out_channels
        if(mid_channels == False):
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, 1, 1, bias=True),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def  __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sampling_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size = 2),
            DoubleConv(in_channels, out_channels),
        )
    def forward(self, x):
        return self.down_sampling_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear = True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, trilinear)
        self.up2 = Up(512, 256 // factor, trilinear)
        self.up3 = Up(256, 128 // factor, trilinear)
        self.up4 = Up(128, 64, trilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        #print(x)
        #print(f"unet x {x.shape}")
        x1 = self.inc(x)
        #print(f"unet x1 {x1.shape}")
        #print(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #print(f"unet x5 {x5.shape}")
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
