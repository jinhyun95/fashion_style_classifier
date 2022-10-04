import torch
import torch.nn as nn
import model.modules as modules
from model.baselines import BaselineClassifier
from model.attentions import ResidualCBAM, ResidualBAM
import numpy as np


class GFLP(nn.Module):
    def __init__(self, args, num_classes):
        super(GFLP, self).__init__()
        base_model = BaselineClassifier(args, num_classes)
        self.args = args
        if args.twophase:
            base_name = {'resnet18': 'ResNet18', 'resnet34': 'ResNet34', 'resnet50': 'ResNet50', 'resnet101': 'ResNet101',
                         'resnext50_32x4d': 'ResNeXt50', 'wide_resnet50_2': 'WideResNet50', 'mobilenet_v2': 'MobileNet'}[args.backbone]
            if args.dataset == 'fashionstyle14' and args.fssplit is not None:
                base_checkpoint = torch.load('checkpoints/%s_split%d_%s.ckpt' % (base_name, args.fssplit, args.dataset))['net_state_dict']
            else:
                base_checkpoint = torch.load('checkpoints/%s_%s.ckpt' % (base_name, args.dataset))['net_state_dict']
            base_checkpoint = {'.'.join(k.split('.')[1:]): base_checkpoint[k] for k in base_checkpoint.keys()}
            base_model.load_state_dict(base_checkpoint, strict=True)
        if args.backbone in ['resnet18', 'resnet34']:
            self.num_features = 512
        elif 'mobilenet' in args.backbone:
            self.num_features = 1280
        else:
            self.num_features = 2048

        feature_extractor = base_model.feature_extractor.net
        if 'mobile' in args.backbone:
            self.feature_extractor_1 = nn.Sequential(*feature_extractor[:15])
        else:
            self.feature_extractor_1 = nn.Sequential(*feature_extractor[:-1])

        # Freeze
        if args.twophase:
            for param in self.feature_extractor_1.parameters():
                param.requires_grad = False
            self.feature_extractor_1.requires_grad = False

        if 'mobile' in args.backbone:
            self.feature_extractor_2 = nn.Sequential(*feature_extractor[15:])
        else:
            self.feature_extractor_2 = feature_extractor[-1]

        self.classifier = list(base_model.classifier.children())[1]
        self.gradcam_module= {}

        self.lambda_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.num_features, self.num_features // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.num_features // 2, self.num_features, 1, bias=False),
            nn.Sigmoid()
        )
        self.normalize = nn.Softmax2d()

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()

        if self.args.twophase:
            self.feature_extractor_1.eval()
            with torch.no_grad():
                feature = self.feature_extractor_1(x)
        else:
            feature = self.feature_extractor_1(x)
        feature = self.feature_extractor_2(feature)
        lambdas = self.lambda_net(feature)
        pooling_weight = self.normalize(feature * lambdas)
        feature = (feature * pooling_weight).sum(-1).sum(-1)
        logits = self.classifier(feature)
        return {'logits': logits, 'gradcam_activation': gradcam_activation, 'visualized_features': visualized_features}


class iSPGaussian(nn.Module):
    def __init__(self, args, num_classes):
        super(iSPGaussian, self).__init__()
        base_model = BaselineClassifier(args, num_classes)
        self.args = args
        if args.twophase:
            base_name = {'resnet18': 'ResNet18', 'resnet34': 'ResNet34', 'resnet50': 'ResNet50', 'resnet101': 'ResNet101',
                         'resnext50_32x4d': 'ResNeXt50', 'wide_resnet50_2': 'WideResNet50', 'mobilenet_v2': 'MobileNet'}[args.backbone]
            if args.dataset == 'fashionstyle14' and args.fssplit is not None:
                base_checkpoint = torch.load('checkpoints/%s_split%d_%s.ckpt' % (base_name, args.fssplit, args.dataset))['net_state_dict']
            else:
                base_checkpoint = torch.load('checkpoints/%s_%s.ckpt' % (base_name, args.dataset))['net_state_dict']
            base_checkpoint = {'.'.join(k.split('.')[1:]): base_checkpoint[k] for k in base_checkpoint.keys()}
            base_model.load_state_dict(base_checkpoint, strict=True)
        if args.backbone in ['resnet18', 'resnet34']:
            self.num_features = 512
        elif 'mobilenet' in args.backbone:
            self.num_features = 1280
        else:
            self.num_features = 2048

        feature_extractor = base_model.feature_extractor.net
        if 'mobile' in args.backbone:
            self.feature_extractor_1 = nn.Sequential(*feature_extractor[:15])
        else:
            self.feature_extractor_1 = nn.Sequential(*feature_extractor[:-1])

        # Freeze
        if args.twophase:
            for param in self.feature_extractor_1.parameters():
                param.requires_grad = False
            self.feature_extractor_1.requires_grad = False

        if 'mobile' in args.backbone:
            self.feature_extractor_2 = nn.Sequential(*feature_extractor[15:])
        else:
            self.feature_extractor_2 = feature_extractor[-1]

        self.classifier = list(base_model.classifier.children())[1]
        self.gradcam_module= {}

        self.eta_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.num_features, self.num_features // 2, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.num_features // 2, self.num_features, 1, bias=True)
        )
        self.normalize = nn.Softmax2d()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()

        if self.args.twophase:
            self.feature_extractor_1.eval()
            with torch.no_grad():
                feature = self.feature_extractor_1(x)
        else:
            feature = self.feature_extractor_1(x)
        feature = self.feature_extractor_2(feature)
        mu_x = self.avg(feature)
        sigma_x = (self.avg(feature.pow(2)) - mu_x.pow(2)).pow(0.5)
        eta = self.eta_net(feature)
        feature = mu_x + eta * sigma_x
        logits = self.classifier(feature.squeeze(-1).squeeze(-1))
        return {'logits': logits, 'gradcam_activation': gradcam_activation, 'visualized_features': visualized_features}


class GLPool(nn.Module):
    def __init__(self, args, num_classes):
        super(GLPool, self).__init__()
        if args.backbone == 'cbam':
            base_model = ResidualCBAM(args, num_classes)
        elif args.backbone == 'bam':
            base_model = ResidualBAM(args, num_classes)
        else:
            base_model = BaselineClassifier(args, num_classes)
        self.args = args
        if args.twophase:
            base_name = {'resnet18': 'ResNet18', 'resnet34': 'ResNet34', 'resnet50': 'ResNet50', 'resnet101': 'ResNet101',
                         'resnext50_32x4d': 'ResNeXt50', 'wide_resnet50_2': 'WideResNet50', 'mobilenet_v2': 'MobileNet',
                         'bam': 'BAM', 'cbam': 'CBAM'}[args.backbone]
            if args.dataset == 'fashionstyle14' and args.fssplit is not None:
                base_checkpoint = torch.load('checkpoints/%s_split%d_%s.ckpt' % (base_name, args.fssplit, args.dataset))['net_state_dict']
            else:
                base_checkpoint = torch.load('checkpoints/%s_%s.ckpt' % (base_name, args.dataset))['net_state_dict']
            base_checkpoint = {'.'.join(k.split('.')[1:]): base_checkpoint[k] for k in base_checkpoint.keys()}
            base_model.load_state_dict(base_checkpoint, strict=True)
        if args.backbone in ['resnet18', 'resnet34']:
            self.num_features = 512
        elif 'mobilenet' in args.backbone:
            self.num_features = 1280
        else:
            self.num_features = 2048

        if args.backbone == 'bam':
            net = base_model.net
            self.feature_extractor_1 = nn.Sequential(
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
            self.feature_extractor_2 = net.layer4
            self.classifier = net.fc
        elif args.backbone == 'cbam':
            net = base_model.net
            self.feature_extractor_1 = nn.Sequential(
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.layer2,
                net.layer3,
            )
            self.feature_extractor_2 = net.layer4
            self.classifier = net.fc
        else:
            feature_extractor = base_model.feature_extractor.net
            if 'mobile' in args.backbone:
                self.feature_extractor_1 = nn.Sequential(*feature_extractor[:15])
            else:
                self.feature_extractor_1 = nn.Sequential(*feature_extractor[:-1])

            if 'mobile' in args.backbone:
                self.feature_extractor_2 = nn.Sequential(*feature_extractor[15:])
            else:
                self.feature_extractor_2 = feature_extractor[-1]

            self.classifier = list(base_model.classifier.children())[1]
        # Freeze
        if args.twophase:
            for param in self.feature_extractor_1.parameters():
                param.requires_grad = False
            self.feature_extractor_1.requires_grad = False
        self.gradients = None
        self.pooling = nn.Conv2d(self.num_features, self.num_features, args.image_size // 32,
                                 groups=self.num_features, bias=False)

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, visualize=False):
        visualized_features = dict()

        if self.args.twophase:
            self.feature_extractor_1.eval()
            with torch.no_grad():
                feature = self.feature_extractor_1(x)
        else:
            feature = self.feature_extractor_1(x)
        feature = self.feature_extractor_2(feature)
        if self.args.mode == 'visualize':
            feature.register_hook(self.save_gradient)
        pooled = self.pooling(feature)
        logits = self.classifier(pooled.squeeze(-1).squeeze(-1))
        if visualize:
            selecteds = pooled.unsqueeze(1).squeeze(-1).squeeze(-1) # B1C
            if isinstance(self.classifier, nn.Sequential):
                theta = torch.stack([list(self.classifier.children())[1].weight.data] * selecteds.size(0), 0)  # BLC
            else:
                theta = torch.stack([self.classifier.weight.data] * selecteds.size(0), 0)  # BLC
            importance = (theta * selecteds) # BLC
            pooling_weight = self.pooling.weight.data
            heatmaps = torch.stack([pooling_weight.view(pooling_weight.size(0), -1)] * x.size(0), 0) # CAWH
            heatmap_mixture = torch.bmm(importance, heatmaps)
            heatmap_mixture = heatmap_mixture.view(x.size(0), heatmap_mixture.size(1), pooling_weight.size(-2), pooling_weight.size(-1))
            for l in range(importance.size(1)):
                visualized_features['label_%d_heatmap' % l] = heatmap_mixture[:, l, :, :]
        return {'logits': logits, 'gradcam_activation': feature, 'visualized_features': visualized_features}


class alphamex(nn.Module):
    def __init__(self, args, num_classes):
        super(alphamex, self).__init__()
        if args.backbone == 'cbam':
            base_model = ResidualCBAM(args, num_classes)
        elif args.backbone == 'bam':
            base_model = ResidualBAM(args, num_classes)
        else:
            base_model = BaselineClassifier(args, num_classes)
        self.args = args
        if args.twophase:
            base_name = \
            {'resnet18': 'ResNet18', 'resnet34': 'ResNet34', 'resnet50': 'ResNet50', 'resnet101': 'ResNet101',
             'resnext50_32x4d': 'ResNeXt50', 'wide_resnet50_2': 'WideResNet50', 'mobilenet_v2': 'MobileNet',
             'bam': 'BAM', 'cbam': 'CBAM'}[args.backbone]
            if args.dataset == 'fashionstyle14' and args.fssplit is not None:
                base_checkpoint = torch.load('checkpoints/%s_split%d_%s.ckpt' % (base_name, args.fssplit, args.dataset))['net_state_dict']
            else:
                base_checkpoint = torch.load('checkpoints/%s_%s.ckpt' % (base_name, args.dataset))['net_state_dict']
            base_checkpoint = {'.'.join(k.split('.')[1:]): base_checkpoint[k] for k in base_checkpoint.keys()}
            base_model.load_state_dict(base_checkpoint, strict=True)
        if args.backbone in ['resnet18', 'resnet34']:
            self.num_features = 512
        elif 'mobilenet' in args.backbone:
            self.num_features = 1280
        else:
            self.num_features = 2048

        if args.backbone == 'bam':
            net = base_model.net
            self.feature_extractor_1 = nn.Sequential(
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
            self.feature_extractor_2 = net.layer4
            self.classifier = net.fc
        elif args.backbone == 'cbam':
            net = base_model.net
            self.feature_extractor_1 = nn.Sequential(
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.layer2,
                net.layer3,
            )
            self.feature_extractor_2 = net.layer4
            self.classifier = net.fc
        else:
            feature_extractor = base_model.feature_extractor.net
            if 'mobile' in args.backbone:
                self.feature_extractor_1 = nn.Sequential(*feature_extractor[:15])
            else:
                self.feature_extractor_1 = nn.Sequential(*feature_extractor[:-1])

            if 'mobile' in args.backbone:
                self.feature_extractor_2 = nn.Sequential(*feature_extractor[15:])
            else:
                self.feature_extractor_2 = feature_extractor[-1]

            self.classifier = list(base_model.classifier.children())[1]
        # Freeze
        if args.twophase:
            for param in self.feature_extractor_1.parameters():
                param.requires_grad = False
            self.feature_extractor_1.requires_grad = False
        self.gradients = None
        self.alpha = nn.Parameter(torch.zeros([1, self.num_features, 1, 1], dtype=torch.float32, requires_grad=True) + 0.65)

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, visualize=False):
        visualized_features = dict()

        if self.args.twophase:
            self.feature_extractor_1.eval()
            with torch.no_grad():
                feature = self.feature_extractor_1(x)
        else:
            feature = self.feature_extractor_1(x)
        feature = self.feature_extractor_2(feature)
        if self.args.mode == 'visualize':
            feature.register_hook(self.save_gradient)
        alpha = self.alpha.clamp(min=0.5 + 1e-12, max=1.0 - 1e-12)
        beta = alpha / (-alpha + 1.)
        alphamex = torch.exp(feature * beta.log()).mean(dim=-1).mean(dim=-1).log() / (beta.squeeze(-1).squeeze(-1).log())
        logits = self.classifier(alphamex)
        if visualize:
            pass
        return {'logits': logits, 'gradcam_activation': feature, 'visualized_features': visualized_features}


class GatedPooling(nn.Module):
    def __init__(self, args, num_classes):
        super(GatedPooling, self).__init__()
        base_model = BaselineClassifier(args, num_classes)
        self.args = args
        if args.twophase:
            base_name = {'resnet18': 'ResNet18', 'resnet34': 'ResNet34', 'resnet50': 'ResNet50', 'resnet101': 'ResNet101',
                         'resnext50_32x4d': 'ResNeXt50', 'wide_resnet50_2': 'WideResNet50', 'mobilenet_v2': 'MobileNet'}[args.backbone]
            if args.dataset == 'fashionstyle14' and args.fssplit is not None:
                base_checkpoint = torch.load('checkpoints/%s_split%d_%s.ckpt' % (base_name, args.fssplit, args.dataset))['net_state_dict']
            else:
                base_checkpoint = torch.load('checkpoints/%s_%s.ckpt' % (base_name, args.dataset))['net_state_dict']
            base_checkpoint = {'.'.join(k.split('.')[1:]): base_checkpoint[k] for k in base_checkpoint.keys()}
            base_model.load_state_dict(base_checkpoint, strict=True)
        if args.backbone in ['resnet18', 'resnet34']:
            self.num_features = 512
        elif 'mobilenet' in args.backbone:
            self.num_features = 1280
        else:
            self.num_features = 2048

        feature_extractor = base_model.feature_extractor.net
        if 'mobile' in args.backbone:
            self.feature_extractor_1 = nn.Sequential(*feature_extractor[:15])
        else:
            self.feature_extractor_1 = nn.Sequential(*feature_extractor[:-1])

        # Freeze
        if args.twophase:
            for param in self.feature_extractor_1.parameters():
                param.requires_grad = False
            self.feature_extractor_1.requires_grad = False

        if 'mobile' in args.backbone:
            self.feature_extractor_2 = nn.Sequential(*feature_extractor[15:])
        else:
            self.feature_extractor_2 = feature_extractor[-1]

        self.classifier = list(base_model.classifier.children())[1]
        self.gradcam_module= {}
        self.alpha = nn.Conv2d(self.num_features, self.num_features, args.image_size // 32,
                               groups=self.num_features, bias=True)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()

        if self.args.twophase:
            self.feature_extractor_1.eval()
            with torch.no_grad():
                feature = self.feature_extractor_1(x)
        else:
            feature = self.feature_extractor_1(x)
        feature = self.feature_extractor_2(feature)
        alpha = self.alpha(feature).sigmoid()
        feature = self.maxpool(feature) * alpha + self.avgpool(feature) * (-alpha + 1.)
        logits = self.classifier(feature.squeeze(-1).squeeze(-1))
        return {'logits': logits, 'gradcam_activation': gradcam_activation, 'visualized_features': visualized_features}


class MixedPooling(nn.Module):
    def __init__(self, args, num_classes):
        super(MixedPooling, self).__init__()
        base_model = BaselineClassifier(args, num_classes)
        self.args = args
        if args.twophase:
            base_name = {'resnet18': 'ResNet18', 'resnet34': 'ResNet34', 'resnet50': 'ResNet50', 'resnet101': 'ResNet101',
                         'resnext50_32x4d': 'ResNeXt50', 'wide_resnet50_2': 'WideResNet50', 'mobilenet_v2': 'MobileNet'}[args.backbone]
            if args.dataset == 'fashionstyle14' and args.fssplit is not None:
                base_checkpoint = torch.load('checkpoints/%s_split%d_%s.ckpt' % (base_name, args.fssplit, args.dataset))['net_state_dict']
            else:
                base_checkpoint = torch.load('checkpoints/%s_%s.ckpt' % (base_name, args.dataset))['net_state_dict']
            base_checkpoint = {'.'.join(k.split('.')[1:]): base_checkpoint[k] for k in base_checkpoint.keys()}
            base_model.load_state_dict(base_checkpoint, strict=True)
        if args.backbone in ['resnet18', 'resnet34']:
            self.num_features = 512
        elif 'mobilenet' in args.backbone:
            self.num_features = 1280
        else:
            self.num_features = 2048

        feature_extractor = base_model.feature_extractor.net
        if 'mobile' in args.backbone:
            self.feature_extractor_1 = nn.Sequential(*feature_extractor[:15])
        else:
            self.feature_extractor_1 = nn.Sequential(*feature_extractor[:-1])

        # Freeze
        if args.twophase:
            for param in self.feature_extractor_1.parameters():
                param.requires_grad = False
            self.feature_extractor_1.requires_grad = False

        if 'mobile' in args.backbone:
            self.feature_extractor_2 = nn.Sequential(*feature_extractor[15:])
        else:
            self.feature_extractor_2 = feature_extractor[-1]

        self.classifier = list(base_model.classifier.children())[1]
        self.gradcam_module= {}
        self.alpha = nn.Parameter(torch.zeros([1, self.num_features, 1, 1], dtype=torch.float32, requires_grad=True))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()

        if self.args.twophase:
            self.feature_extractor_1.eval()
            with torch.no_grad():
                feature = self.feature_extractor_1(x)
        else:
            feature = self.feature_extractor_1(x)
        feature = self.feature_extractor_2(feature)
        alpha = self.alpha.sigmoid()
        feature = self.maxpool(feature) * alpha + self.avgpool(feature) * (-alpha + 1.)
        logits = self.classifier(feature.squeeze(-1).squeeze(-1))
        return {'logits': logits, 'gradcam_activation': gradcam_activation, 'visualized_features': visualized_features}
