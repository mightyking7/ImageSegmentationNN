"""
Models that make up Unet
"""

"""
    data aumentation, flip and crop
    tiling, up conv first and if higher res needed do dilation
    don't understand copy & crop b/c it's not talked about in the paper
    outputing integer semantic label
    how to use color_map.json
    
    padd conv for no cropping, add skip connection for copy
    
    for every up conve concat feature from lower layer or add if they have the same channels
    
    build color from integer labels using color map
"""

import torch
import torch.nn as nn
import torch.functional as F

class DoubleConv(nn.Module):

    def __init__(self, inChannels, outChannels):
        """
            Double conv layer used in architecture
            :param inChannels: input channels
            :param outChannels: output channels
        """
        super(DoubleConv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convs(x)

class DownConv(nn.Module):

    def __init__(self, inChannels, outChannels):
        super(DownConv, self).__init__()
        self.convs = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(inChannels, outChannels)
        )

    def forward(self, x):
        return self.convs(x)

class UpConv(nn.Module):

    def __init__(self, inChannles, outChannels, bilinear=True):
        super(UpConv, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(inChannles // 2, outChannels // 2, 2, stride=2)

        self.conv = DoubleConv(inChannles, outChannels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, inChannels, outChannels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, 1)

    def foward(self, x):
        return OutConv(x)

