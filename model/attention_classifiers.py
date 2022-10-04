import torch
import torch.nn as nn
import model.modules as modules
import numpy as np


class MixedPoolingAttention(nn.Module):
    def __init__(self, nchannel, out=1, grouped=False):
        super(MixedPoolingAttention, self).__init__()
        group = out if grouped else 1
        self.preconv = nn.Conv2d(nchannel, nchannel, kernel_size=1, bias=False, groups=group)
        self.prebn = nn.BatchNorm2d(nchannel)
        self.relu = nn.ReLU()
        if group > 1:
            self.conv = nn.ModuleList(
                [nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, stride=2, bias=False)
                 for _ in range(out)])
        else:
            self.conv = nn.Conv2d(in_channels=2, out_channels=out, kernel_size=7, padding=3, stride=2, bias=False)

    def forward(self, x):
        x = self.relu(self.prebn(self.preconv(x)))
        if isinstance(self.conv, nn.ModuleList):
            x = torch.split(x, x.size(1) // len(self.conv), dim=1)
            x = torch.cat(
                [self.conv[i](torch.cat([x[i].max(dim=1, keepdim=True)[0], x[i].mean(dim=1, keepdim=True)], dim=1)) for
                 i in range(len(self.conv))], dim=1)
        else:
            x = self.conv(torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1))
        return x


class BottleneckAttention(nn.Module):
    def __init__(self, nchannel, image_size, where, out=1, grouped=False):
        super(BottleneckAttention, self).__init__()
        group = out if grouped else 1
        modules = [nn.Conv2d(nchannel, nchannel, kernel_size=1, groups=group),
                   nn.BatchNorm2d(nchannel),
                   nn.ReLU()]
        for i in range(5 - where):
            modules.append(nn.MaxPool2d(3, 2, 1))
            modules.append(nn.Conv2d(nchannel // (2 ** i), nchannel // (2 ** (i + 1)), kernel_size=1, groups=group, bias=False))
            modules.append(nn.BatchNorm2d(nchannel // (2 ** (i + 1))))
            modules.append(nn.ReLU())
        modules.append(nn.UpsamplingBilinear2d(size=(image_size // (2 ** (2 +  where)), image_size // (2 ** (2 +  where)))))
        modules.append(nn.Conv2d(nchannel // (2 ** (5 - where)), out, kernel_size=1, groups=group, bias=False))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


class ConvAttention(nn.Module):
    def __init__(self, nchannel, out=1, grouped=False, layers=5):
        super(ConvAttention, self).__init__()
        group = out if grouped else 1
        modules = []
        for i in range(layers - 1):
            modules.append(nn.Conv2d(nchannel, nchannel, kernel_size=3, padding=1, groups=group, bias=False))
            modules.append(nn.BatchNorm2d(nchannel)),
            modules.append(nn.ReLU())
        modules.append(nn.Conv2d(nchannel, out, kernel_size=3, padding=1, stride=2, groups=group, bias=False))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


class BivariateNormalAttention(nn.Module):
    def __init__(self, nchannel, sigma_ratio=1., out=1, grouped=False, gmm=1):
        super(BivariateNormalAttention, self).__init__()
        group = out if grouped else 1
        self.sigma_ratio = sigma_ratio
        self.conv = nn.Sequential(nn.Conv2d(nchannel, nchannel // 2, kernel_size=3, padding=1, groups=group, bias=False),
                                  nn.BatchNorm2d(nchannel // 2),
                                  nn.ReLU(),
                                  nn.Conv2d(nchannel // 2, nchannel // 2, kernel_size=3, padding=1, groups=group, bias=False),
                                  nn.BatchNorm2d(nchannel // 2),
                                  nn.ReLU(),
                                  nn.AdaptiveAvgPool2d((7, 7)),
                                  nn.Conv2d(nchannel // 2, nchannel // 4, kernel_size=3, padding=1, groups=group, bias=False),
                                  nn.BatchNorm2d(nchannel // 4),
                                  nn.ReLU(),
                                  nn.Conv2d(nchannel // 4, nchannel // 4, kernel_size=3, padding=1, groups=group, bias=False),
                                  nn.BatchNorm2d(nchannel // 4),
                                  nn.ReLU(),
                                  nn.AdaptiveAvgPool2d((3, 3)),
                                  nn.Conv2d(nchannel // 4, nchannel // 8, kernel_size=3, padding=1, groups=group, bias=False),
                                  nn.BatchNorm2d(nchannel // 8),
                                  nn.ReLU()
                                  )
        self.gmm = gmm
        self.fc = nn.ModuleList([nn.Linear(9 * nchannel // (8 * group), 4 * gmm * (out // group), bias=False)
                                 for _ in range(group)])
        self.mixture_weights = nn.Parameter(torch.zeros((1, out, gmm, 1, 1), requires_grad=True))

    def forward(self, x):
        sizes = x.size()
        x = self.conv(x).flatten(start_dim=1)
        x = torch.split(x, x.size(1) // len(self.fc), dim=1)
        x = torch.cat([self.fc[i](x[i]).view(sizes[0], -1, self.gmm, 4) for i in range(len(x))], 1)
        R = 3.
        m_x = x[:, :, :, 0].sigmoid() * float(sizes[2] // 2 - 1)
        m_y = x[:, :, :, 1].sigmoid() * float(sizes[3] // 2 - 1)
        r = ((x[:, :, :, 2].sigmoid() * 2. - 1.) * np.log(R)).exp()
        rho = x[:, :, :, 3].sigmoid() * 1.6 - 0.8
        sigma = float(sizes[2]) / float(2 * self.sigma_ratio)
        attention = []
        denom = - 0.5 * (1 - rho.pow(2)).pow(-1) / sigma ** 2
        for i in range(sizes[2] // 2):
            y_axis_attention = []
            for j in range(sizes[3] // 2):
                y_axis_attention.append((denom * ((m_x - i).pow(2) * r + (m_y - j).pow(2) / r - 2 * rho * (m_x - i) * (m_y - j))).exp())
            attention.append(torch.stack(y_axis_attention, -1))
        attention = torch.stack(attention, dim=-2)
        attention /= attention.sum(dim=3, keepdim=True).sum(dim=4, keepdim=True)
        mixture_weight = torch.softmax(self.mixture_weights, dim=2)
        attention = (attention * mixture_weight).sum(dim=2, keepdims=False)
        return attention


class PooledAttentionAggregation(nn.Module):
    def __init__(self, args, num_classes):
        super(PooledAttentionAggregation, self).__init__()
        backbone = modules.Backbone(args)
        self.args = args
        if args.backbone in ['resnet18', 'resnet34'] or 'vgg' in args.backbone:
            self.num_features = 512
        else:
            self.num_features = 2048
        if 'resnet' in args.backbone or 'resnext' in args.backbone or 'resnest' in args.backbone:
            backbone = list(backbone.net.children())
            self.pre_attention = nn.Sequential(*backbone[:-1])

            if 'bottleneck' in args.exp_name:
                self.attention = BottleneckAttention(self.num_features // 2, args.image_size, 3, args.attentions, False)
            elif 'pooling' in args.exp_name:
                self.attention = MixedPoolingAttention(self.num_features // 2, args.attentions, False)
            elif 'normal' in args.exp_name or 'Gaussian' in args.exp_name:
                self.attention = BivariateNormalAttention(self.num_features // 2, args.sigma_ratio, args.attentions, False, args.gmm)
            else:
                self.attention = ConvAttention(self.num_features // 2, args.attentions, False)

            if 'softmax' in args.exp_name:
                self.normalizer = 'softmax'
            else:
                self.normalizer = 'sigmoid'

            self.trunk_branch = backbone[-1]
            self.pooling = modules.GlobalAvgPooling()
            self.classifier = nn.Linear(self.num_features, num_classes)
        else:
            raise NotImplementedError
        if hasattr(list(self.trunk_branch.children())[-1], 'conv3'):
            self.gradcam_module= {'last_conv': list(self.trunk_branch.children())[-1].conv3}
        else:
            self.gradcam_module = {'last_conv': list(self.trunk_branch.children())[-1].conv2}

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()
        x = self.pre_attention(x)
        if isinstance(self.attention, BivariateNormalAttention):
            attention = self.attention(x)
        elif self.normalizer == 'softmax':
            attention = self.attention(x).flatten(1).softmax(1).view([x.size(0), -1, x.size(2) // 2, x.size(3) // 2])
        else:
            attention = self.attention(x).sigmoid()
        x = self.trunk_branch(x)
        gradcam_activation['last_conv'] = x
        if self.args.res:
            x = torch.stack([self.pooling(x)] + [(x * attention[:, a, :, :].unsqueeze(1)).sum(dim=[2, 3]) for a in range(attention.size(1))], -1).max(dim=-1, keepdim=False)[0]
        else:
            x = torch.stack([(x * attention[:, a, :, :].unsqueeze(1)).sum(dim=[2, 3]) for a in range(attention.size(1))], -1).max(dim=-1, keepdim=False)[0]
        for i in range(self.args.attentions):
            visualized_features['attention_%d' % (i + 1)] = attention[:, i, :, :].squeeze(1)
        logits = self.classifier(x)
        return {'logits': logits, 'gradcam_activation': gradcam_activation,
                'visualized_features': visualized_features, 'attentions': [a.flatten(1) for a in attention]}


class GroupedAttention(nn.Module):
    def __init__(self, args, num_classes):
        super(GroupedAttention, self).__init__()
        backbone = modules.Backbone(args)
        self.args = args
        if args.backbone in ['resnet18', 'resnet34'] or 'vgg' in args.backbone:
            num_features = 512
        else:
            num_features = 2048
        if 'resnet' in args.backbone or 'resnext' in args.backbone or 'resnest' in args.backbone:
            backbone = list(backbone.net.children())
            self.pre_attention = nn.Sequential(*backbone[:4 + args.where])
            if 'bottleneck' in args.exp_name:
                self.attention = BottleneckAttention(num_features // (2 ** (4 - args.where)), args.image_size, args.where, args.attentions, False)
            elif 'pooling' in args.exp_name:
                self.attention = MixedPoolingAttention(num_features // (2 ** (4 - args.where)), args.attentions, False)
            elif 'normal' in args.exp_name or 'Gaussian' in args.exp_name:
                self.attention = BivariateNormalAttention(num_features // (2 ** (4 - args.where)), args.sigma_ratio, args.attentions, False, args.gmm)
            else:
                self.attention = ConvAttention(num_features // (2 ** (4 - args.where)), args.attentions, False)
            if 'softmax' in args.exp_name:
                self.normalizer = 'softmax'
            else:
                self.normalizer = 'sigmoid'
            self.trunk_branch = backbone[4 + args.where]
            self.post_attention = nn.Sequential(*backbone[5 + args.where:])
            self.pooling = modules.GlobalAvgPooling()
            self.classifier = nn.Linear(num_features, num_classes)
        else:
            raise NotImplementedError
        if args.where == 3:
            self.gradcam_module = {'last_conv': list(self.trunk_branch.children())[-1].conv3}
        else:
            self.gradcam_module = {'last_conv': list(list(self.post_attention.children())[-1].children())[-1].conv3}

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()
        x = self.pre_attention(x)
        if isinstance(self.attention, BivariateNormalAttention):
            attention = self.attention(x)
        elif self.normalizer == 'softmax':
            attention = self.attention(x).flatten(1).softmax(1).view([x.size(0), -1, x.size(2) // 2, x.size(3) // 2])
        else:
            attention = self.attention(x).sigmoid()
        expanded_attention = torch.cat([torch.stack([attention[:, i, :, :]] * (2 * x.size(1) // self.args.attentions), 1) for i in range(attention.size(1))], 1)
        x = self.trunk_branch(x)
        gradcam_activation['last_conv'] = x
        if self.args.res:
            x = x * (expanded_attention + 1.)
        else:
            x = x * expanded_attention
        x = self.post_attention(x)
        for i in range(self.args.attentions):
            visualized_features['attention_%d' % (i + 1)] = attention[:, i, :, :].squeeze(1)

        return {'logits': self.classifier(self.pooling(x)), 'gradcam_activation': gradcam_activation,
                'visualized_features': visualized_features, 'attentions': [a.flatten(1) for a in attention]}


class ExplicitGroupedAttention(nn.Module):
    def __init__(self, args, num_classes):
        super(ExplicitGroupedAttention, self).__init__()
        backbone = modules.Backbone(args)
        self.args = args
        if args.backbone in ['resnet18', 'resnet34'] or 'vgg' in args.backbone:
            num_features = 512
        else:
            num_features = 2048
        if 'resnet' in args.backbone or 'resnext' in args.backbone or 'resnest' in args.backbone:
            backbone = list(backbone.net.children())
            self.pre_attention = nn.Sequential(*backbone[:4 + args.where])
            if 'bottleneck' in args.exp_name:
                self.attention = BottleneckAttention(num_features // (2 ** (4 - args.where)), args.image_size, args.where, args.attentions, True)
            elif 'pooling' in args.exp_name:
                self.attention = MixedPoolingAttention(num_features // (2 ** (4 - args.where)), args.attentions, True)
            elif 'normal' in args.exp_name or 'Gaussian' in args.exp_name:
                self.attention = BivariateNormalAttention(num_features // (2 ** (4 - args.where)), args.sigma_ratio, args.attentions, True, args.gmm)
            else:
                self.attention = ConvAttention(num_features // (2 ** (4 - args.where)), args.attentions, True)
            if 'softmax' in args.exp_name:
                self.normalizer = 'softmax'
            else:
                self.normalizer = 'sigmoid'
            self.trunk_branch = backbone[4 + args.where]
            self.post_attention = nn.Sequential(*backbone[5 + args.where:])
            self.pooling = modules.GlobalAvgPooling()
            self.classifier = nn.Linear(num_features, num_classes)
        else:
            raise NotImplementedError
        if args.where == 3:
            self.gradcam_module = {'last_conv': list(self.trunk_branch.children())[-1].conv3}
        else:
            self.gradcam_module = {'last_conv': list(list(self.post_attention.children())[-1].children())[-1].conv3}

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()
        x = self.pre_attention(x)
        if isinstance(self.attention, BivariateNormalAttention):
            attention = self.attention(x)
        elif self.normalizer == 'softmax':
            attention = self.attention(x).flatten(1).softmax(1).view([x.size(0), -1, x.size(2) // 2, x.size(3) // 2])
        else:
            attention = self.attention(x).sigmoid()
        expanded_attention = torch.cat([torch.stack([attention[:, i, :, :]] * (2 * x.size(1) // self.args.attentions), 1) for i in range(attention.size(1))], 1)
        x = self.trunk_branch(x)
        gradcam_activation['last_conv'] = x
        if self.args.res:
            x = x * (expanded_attention + 1.)
        else:
            x = x * expanded_attention
        x = self.post_attention(x)
        for i in range(self.args.attentions):
            visualized_features['attention_%d' % (i + 1)] = attention[:, i, :, :].squeeze(1)

        return {'logits': self.classifier(self.pooling(x)), 'gradcam_activation': gradcam_activation,
                'visualized_features': visualized_features, 'attentions': [a.flatten(1) for a in attention]}


class AttentionPerClass(nn.Module):
    def __init__(self, args, num_classes):
        super(AttentionPerClass, self).__init__()
        assert 'resnet' in args.backbone or 'resnext' in args.backbone or 'resnest' in args.backbone
        backbone = modules.Backbone(args)
        self.args = args
        if args.backbone in ['resnet18', 'resnet34']:
            self.num_features = 512
        else:
            self.num_features = 2048
        self.feat_per_class = args.features_per_class
        self.num_classes = num_classes
        backbone = list(backbone.net.children())
        self.pre_attention = nn.Sequential(*backbone[:-1])
        self.attention = ConvAttention(self.num_features // 2, num_classes, False, args.attention_layers)
        self.trunk_branch = backbone[-1]
        self.channel_matching = nn.Conv2d(self.num_features, self.feat_per_class * num_classes, 1, bias=False)
        self.pooling = modules.GlobalAvgPooling()
        self.classifier = nn.Linear(self.feat_per_class, 1)
        self.gradcam_module = {'last_conv': self.channel_matching}

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()
        x = self.pre_attention(x)
        attention = self.attention(x)
        attention = torch.flatten(attention, start_dim=2).softmax(2).view(attention.size())
        for i in range(attention.size(1)):
            visualized_features['attention_class_%d' % i] = attention[:, i, :, :].squeeze(1)
        x = self.trunk_branch(x)
        x = self.channel_matching(x)
        gradcam_activation['last_conv'] = x
        classwise_features = torch.split(x, split_size_or_sections=self.feat_per_class, dim=1)
        local_feature = [classwise_features[i] * attention[:, i : i + 1, :, :] for i in range(self.num_classes)]
        local_feature = [local_feature[i].sum(3).sum(2) for i in range(self.num_classes)]
        global_feature = [self.pooling(classwise_features[i]) for i in range(self.num_classes)]
        merged_feature = [torch.where(local_feature[i] > global_feature[i], local_feature[i], global_feature[i])
                          for i in range(self.num_classes)]
        logits = torch.cat([self.classifier(f) for f in merged_feature], dim=1)
        return {'logits': logits, 'gradcam_activation': gradcam_activation, 'visualized_features': visualized_features}
