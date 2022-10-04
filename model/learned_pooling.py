import torch
import torchvision
import torch.nn as nn
import model.modules as modules
from model.baselines import BaselineClassifier
import numpy as np
from model.attentions import ResidualBAM, ResidualCBAM


class GaussianAttention(nn.Module):
    def __init__(self, nchannel, sigma_ratio=1., out=1, layers=3, resolution=7):
        super(GaussianAttention, self).__init__()
        self.sigma_ratio = sigma_ratio
        if layers == 2:
            self.conv = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)),
                                      nn.Conv2d(nchannel, nchannel // 4, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 4),
                                      nn.ReLU(),
                                      nn.Conv2d(nchannel // 4, nchannel // 16, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 16),
                                      nn.ReLU()
                                      )
        elif layers == 3:
            self.conv = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)),
                                      nn.Conv2d(nchannel, nchannel, kernel_size=1, padding=0, bias=False),
                                      nn.BatchNorm2d(nchannel),
                                      nn.ReLU(),
                                      nn.Conv2d(nchannel, nchannel // 4, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 4),
                                      nn.ReLU(),
                                      nn.Conv2d(nchannel // 4, nchannel // 16, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 16),
                                      nn.ReLU()
                                      )
        elif layers == 4:
            self.conv = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)),
                                      nn.Conv2d(nchannel, nchannel // 2, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 2),
                                      nn.ReLU(),
                                      nn.Conv2d(nchannel // 2, nchannel // 4, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 4),
                                      nn.ReLU(),
                                      nn.Conv2d(nchannel // 4, nchannel // 8, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 8),
                                      nn.ReLU(),
                                      nn.Conv2d(nchannel // 8, nchannel // 16, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 16),
                                      nn.ReLU()
                                      )
        elif layers == 5:
            self.conv = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)),
                                      nn.Conv2d(nchannel, nchannel, kernel_size=1, padding=0, bias=False),
                                      nn.BatchNorm2d(nchannel),
                                      nn.ReLU(),
                                      nn.Conv2d(nchannel, nchannel // 2, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 2),
                                      nn.ReLU(),
                                      nn.Conv2d(nchannel // 2, nchannel // 4, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 4),
                                      nn.ReLU(),
                                      nn.Conv2d(nchannel // 4, nchannel // 8, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 8),
                                      nn.ReLU(),
                                      nn.Conv2d(nchannel // 8, nchannel // 16, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(nchannel // 16),
                                      nn.ReLU()
                                      )
        self.natt = out
        self.fc = nn.Linear(49 * nchannel // 16, 4 * out, bias=False)
        self.resolution = resolution

    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        x = x * x.size(1) ** 0.5 / torch.norm(x, dim=1, keepdim=True)
        x = self.fc(x).view(-1, self.natt, 4)
        R = 2.
        m_x = x[:, :, 0].sigmoid() * float(self.resolution) - 0.5
        m_y = x[:, :, 1].sigmoid() * float(self.resolution) - 0.5
        r = ((x[:, :, 2].sigmoid() * 2. - 1.) * np.log(R)).exp()
        rho = x[:, :, 3].sigmoid() * 1.8 - 0.9
        sigma = float(self.resolution) / float(self.sigma_ratio)
        attention = []
        denom = - 0.5 * (1 - rho.pow(2)).pow(-1) / (sigma ** 2 + 49e-12)
        for i in range(self.resolution):
            y_axis_attention = []
            for j in range(self.resolution):
                y_axis_attention.append((denom * ((m_x - i).pow(2) * r + (m_y - j).pow(2) / r - 2 * rho * (m_x - i) * (m_y - j))).exp() + 1e-12)
            attention.append(torch.stack(y_axis_attention, -1))
        attention = torch.stack(attention, dim=-2)
        attention /= attention.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        return attention


class LearnedPooling(nn.Module):
    def __init__(self, args, num_classes):
        super(LearnedPooling, self).__init__()
        if args.backbone == 'cbam':
            base_model = ResidualCBAM(args, num_classes)
        elif args.backbone == 'bam':
            base_model = ResidualBAM(args, num_classes)
        else:
            base_model = BaselineClassifier(args, num_classes)
        base_name = {'resnet18': 'ResNet18', 'resnet34': 'ResNet34', 'resnet50': 'ResNet50', 'resnet101': 'ResNet101',
                     'resnext50_32x4d': 'ResNeXt50', 'wide_resnet50_2': 'WideResNet50', 'mobilenet_v2': 'MobileNet',
                     'bam': 'BAM', 'cbam': 'CBAM'}[args.backbone]
        if args.dataset == 'fashionstyle14' and args.fssplit is not None:
            base_checkpoint = torch.load('checkpoints/%s_split%d_%s.ckpt' % (base_name, args.fssplit, args.dataset))['net_state_dict']
        else:
            base_checkpoint = torch.load('checkpoints/%s_%s.ckpt' % (base_name, args.dataset))['net_state_dict']
        base_checkpoint = {'.'.join(k.split('.')[1:]): base_checkpoint[k] for k in base_checkpoint.keys()}
        base_model.load_state_dict(base_checkpoint, strict=True)
        self.args = args

        if args.trunk == 't':
            if args.backbone in ['resnet18', 'resnet34']:
                self.num_features = 256
            elif 'mobilenet' in args.backbone:
                self.num_features = 160
            else:
                self.num_features = 1024
        else:
            if args.backbone in ['resnet18', 'resnet34']:
                self.num_features = 512
            elif 'mobilenet' in args.backbone:
                self.num_features = 1280
            else:
                self.num_features = 2048

        if args.backbone == 'bam':
            net = base_model.net
            self.pre_attention = nn.Sequential(
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.bam1,
                net.layer2,
                net.bam2,
                net.layer3,
                net.bam3
            )
            self.trunk_branch = net.layer4
            self.classifier = net.fc
        elif args.backbone == 'cbam':
            net = base_model.net
            self.pre_attention = nn.Sequential(
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.layer2,
                net.layer3,
            )
            self.trunk_branch = net.layer4
            self.classifier = net.fc
        else:
            feature_extractor = list(base_model.feature_extractor.net.children())
            if 'mobile' in args.backbone:
                self.pre_attention = nn.Sequential(*feature_extractor[:15])
            else:
                self.pre_attention = nn.Sequential(*feature_extractor[:-1])
            if 'mobile' in args.backbone:
                self.trunk_branch = nn.Sequential(*feature_extractor[15:])
            else:
                self.trunk_branch = feature_extractor[-1]
            self.classifier = list(base_model.classifier.children())[1]

        # Freeze
        for param in self.pre_attention.parameters():
            param.requires_grad = False
        self.pre_attention.requires_grad = False
        if args.trunk == 'p':
            for param in self.trunk_branch.parameters():
                param.requires_grad = False

        self.attention = GaussianAttention(self.num_features, args.sigma_ratio, args.attentions, args.layers,
                                           resolution=args.image_size // 32)
        self.pooling = modules.GlobalAvgPooling()
        self.gradients = None
        # last = list(self.trunk_branch.children())[-1]
        # if isinstance(last, torchvision.models.resnet.Bottleneck):
        #     self.gradcam_module['last_conv'] = last.conv3
        # elif isinstance(last, torchvision.models.resnet.BasicBlock):
        #     self.gradcam_module['last_conv'] = last.conv2
        # elif 'mobile' in args.backbone:
        #     self.gradcam_module['last_conv'] = list(last.children())[0]
        self.register_buffer('stochastic_pooling', torch.tensor([1.0]))

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, visualize=False):
        visualized_features = dict()

        with torch.no_grad():
            self.pre_attention.eval()
            x = self.pre_attention(x)
        if self.args.trunk == 't':
            attention = self.attention(x)
        if self.args.trunk == 'p':
            with torch.no_grad():
                self.trunk_branch.eval()
                x = self.trunk_branch(x)
        else:
            x = self.trunk_branch(x)
        if self.args.trunk != 't':
            attention = self.attention(x)
        for i in range(self.args.attentions):
            visualized_features['attention_%d' % (i + 1)] = attention[:, i, :, :]
        if self.args.mode == 'visualize':
            x.register_hook(self.save_gradient)
        cat = torch.stack([self.pooling(x)] +
                          [(x * attention[:, a, :, :].unsqueeze(1)).sum(dim=[2, 3])for a in range(attention.size(1))],
                          -1)
        f = cat.max(dim=-1, keepdim=False)[0]

        # if label is not None and visualize:
        #     selecteds = (cat == f.unsqueeze(-1)).to(torch.float32)  # BCA
        #     selecteds = selecteds / selecteds.sum(dim=-1, keepdim=True) * f.unsqueeze(-1)
        #     if isinstance(self.classifier, nn.Sequential):
        #         theta = torch.stack([list(self.classifier.children())[1].weight.data] * selecteds.size(0), 0)  # BLC
        #     else:
        #         theta = torch.stack([self.classifier.weight.data] * selecteds.size(0), 0)  # BLC
        #     importance = torch.bmm(theta, selecteds).clamp(min=0.)  # BLA
        #     if len(label.size()) == 2:
        #         importance = (importance * label.unsqueeze(2)).sum(1) # BA
        #     else:
        #         importance = (importance * torch.eye(theta.size(1))[label].to(theta.device).unsqueeze(2)).sum(1)
        #     importance = (importance / importance.sum(dim=1, keepdim=True)).unsqueeze(2).unsqueeze(3)
        #     heatmaps = torch.cat([torch.ones_like(attention[:, :1, :, :]) / attention.size(2) / attention.size(3), attention], 1)
        #     visualized_features['aggregated_heatmap'] = (heatmaps * importance).sum(dim=1)
        if visualize:
            selecteds = (cat == f.unsqueeze(-1)).to(torch.float32) # BCA
            selecteds = selecteds / selecteds.sum(dim=-1, keepdim=True)
            if isinstance(self.classifier, nn.Sequential):
                theta = torch.stack([list(self.classifier.children())[1].weight.data] * selecteds.size(0), 0)  # BLC
            else:
                theta = torch.stack([self.classifier.weight.data] * selecteds.size(0), 0)  # BLC
            importance = torch.bmm((theta * f.unsqueeze(1)).clamp(min=0.), selecteds).unsqueeze(3).unsqueeze(4) # BLAWH
            heatmaps = torch.cat([torch.ones_like(attention[:, :1, :, :]) / attention.size(2) / attention.size(3), attention], 1)
            for l in range(importance.size(1)):
                visualized_features['label_%d_heatmap' % l] = (heatmaps * importance[:, l, :, :, :]).sum(dim=1)
        logits = self.classifier(f)
        return {'logits': logits, 'gradcam_activation': x, 'visualized_features': visualized_features}
