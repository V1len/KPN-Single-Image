import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicNet(nn.Module):
    def __init__(self, in_channel, out_channel, g=16, channel_att=False, spatial_att=False):
        super(BasicNet, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.ReLU(),
        )
        if channel_att:
            self.att_channel = nn.Sequential(
                nn.Conv2d(2*out_channel, out_channel//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_channel//g, out_channel, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_spatial = nn.Sequential(
                nn.Conv2d(2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

