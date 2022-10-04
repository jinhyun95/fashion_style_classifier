import torch
import torch.nn as nn
from torchvision import models

class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        net = getattr(models, args.backbone)(pretrained=True)
        if 'resnet' in args.backbone or 'resnext' in args.backbone:
            self.net = nn.Sequential(*list(net.children())[:-2])
        elif 'mobilenet' in args.backbone:
            self.net = net.features
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.net(x)

class GlobalMaxPooling(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()

    def forward(self, x, mask=None):
        # mask: B, H, W, [0, 1]
        if mask is not None:
            x = x * mask.unsqueeze(1)
        return x.max(dim=-1, keepdim=False)[0].max(dim=-1, keepdim=False)[0]


class GlobalAvgPooling(nn.Module):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()

    def forward(self, x, mask=None):
        # mask: B, H, W, [0, 1]
        if mask is not None:
            x = x * mask.unsqueeze(1)
            return x.sum(dim=-1, keepdim=False).sum(dim=-1, keepdim=False) / mask.sum()
        else:
            return x.mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False)


class Flatten(nn.Module):
    def __init__(self, dim):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.flatten(x, self.dim)


# Following attention modules are implementation of CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).
class ChannelAttentionCBAM(nn.Module):
    def __init__(self, n_channels, reduction_ratio=16):
        super(ChannelAttentionCBAM, self).__init__()
        self.n_channels = n_channels
        self.mlp = nn.Sequential(
            Flatten(1),
            nn.Linear(n_channels, n_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(n_channels // reduction_ratio, n_channels)
            )
        self.avg_pool = GlobalAvgPooling()
        self.max_pool = GlobalMaxPooling()
        self.bn = nn.BatchNorm1d(n_channels)
        for key in self.bn.state_dict().keys():
            self.bn.state_dict()[key][...] = 0.

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        # channel_attention = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).unsqueeze(2).unsqueeze(3)
        channel_attention = torch.sigmoid(self.bn(self.mlp(avg_pool) + self.mlp(max_pool))).unsqueeze(2).unsqueeze(3)

        return channel_attention


class SpatialAttentionCBAM(nn.Module):
    def __init__(self):
        super(SpatialAttentionCBAM, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        for key in self.bn.state_dict().keys():
            self.bn.state_dict()[key][...] = 0.

    def forward(self, x):
        avg_pool = x.mean(1)
        max_pool = x.max(1)[0]
        spatial_attention = torch.sigmoid(self.bn(self.conv(torch.stack([avg_pool, max_pool], 1))))
        return spatial_attention


class SimpleGroupSpatialAttention(nn.Module):
    def __init__(self, channels, group):
        super(SimpleGroupSpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=group, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )
        self.group = group
        self.channel_per_group = channels // group

    def forward(self, x):
        spatial_attention = self.conv(x).sigmoid()
        return torch.cat([torch.cat([f] * self.channel_per_group, 1) for f in spatial_attention], 1)


class BottleneckSpatialAttentionDownsample(nn.Module):
    def __init__(self, channels, group, n_downsample):
        super(BottleneckSpatialAttentionDownsample, self).__init__()
        self.group = group if group is not None else channels
        self.channel_per_group = channels // self.group
        downsample = []
        grouped = [nn.Conv2d(in_channels=channels, out_channels=self.group, kernel_size=7, padding=3, bias=False),
                   nn.BatchNorm2d(num_features=self.group, eps=1e-5, momentum=0.01, affine=True)]
        upsample = []
        for _ in range(n_downsample):
            downsample.extend(
                [nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=7, padding=3, bias=False),
                 nn.BatchNorm2d(num_features=channels, eps=1e-5, momentum=0.01, affine=True),
                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
            upsample.extend(
                [nn.UpsamplingBilinear2d(scale_factor=2),
                 nn.Conv2d(in_channels=self.group, out_channels=self.group, kernel_size=7, padding=3, bias=False),
                 nn.BatchNorm2d(num_features=self.group, eps=1e-5, momentum=0.01, affine=True)])

        for m in downsample + grouped + upsample:
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        self.attention = nn.Sequential(*(downsample + grouped + upsample))

    def forward(self, x):
        spatial_attention = self.attention(x)
        return torch.cat([spatial_attention] * self.channel_per_group, 1)

