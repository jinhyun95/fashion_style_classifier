import torch
import torch.nn as nn
import model.modules as modules

class LocalizedGlobalPoolingClassifier(nn.Module):
    def __init__(self, args, num_classes):
        super(LocalizedGlobalPoolingClassifier, self).__init__()
        self.feature_extractor = modules.Backbone(args)
        if args.backbone in ['resnet18', 'resnet34'] or 'vgg' in args.backbone:
            num_features = 512
        else:
            num_features = 2048
        if 'resnet' in args.backbone or 'resnext' in args.backbone or 'resnest' in args.backbone:
            self.classifier = nn.Linear(num_features, num_classes)
        else:
            raise NotImplementedError
        self.gradcam_module = dict()
        for module in list(self.feature_extractor.modules())[::-1]:
            if isinstance(module, nn.Conv2d):
                self.gradcam_module['last_conv'] = module
                break
        self.maxidx = nn.AdaptiveMaxPool2d((1, 1), return_indices=True)
        gridsize = args.image_size // 32
        grid = torch.zeros((gridsize, gridsize, 2)).to(torch.int64)
        for i in range(gridsize):
            grid[:, i, 0] = torch.arange(gridsize)
            grid[i, :, 1] = torch.arange(gridsize)
        # 1, 1, H, W, 2
        grid = grid.to(torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('grid', grid)

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()
        feature = self.feature_extractor(x)
        gradcam_activation['last_conv'] = feature
        maxval, idxs = self.maxidx(feature)
        idxs = idxs[:, :, 0, 0].to(torch.int64)
        # B, C, H, W, 2
        idxs = torch.stack([torch.floor_divide(idxs, feature.size(3)).to(torch.int64),
                            torch.remainder(idxs, feature.size(3)).to(torch.int64)], dim=-1).unsqueeze(2).unsqueeze(3)
        idxs  = idxs.expand(feature.size(0), feature.size(1), feature.size(2), feature.size(3), 2).to(torch.float32)
        distance = (idxs - self.grid).pow(2).sum(dim=-1, keepdim=False).pow(1/2) + 1e-8 # B, C, H, W
        difference = maxval.detach() - feature.detach()
        slope = difference.div(distance)
        neighbor = (distance < 2).to(torch.float32)
        slope = ((slope * neighbor).sum(dim=3).sum(dim=2) / neighbor.sum(dim=3).sum(dim=2)).unsqueeze(2).unsqueeze(3)
        weight = torch.clamp(maxval.detach() - distance * slope, min=0.)
        weight = (weight + 1e-8) / (weight + 1e-8).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        for i in range(10):
            visualized_features['weight%d' % i] = weight[:, i, :, :]
        feature = (feature * weight).sum(dim=3).sum(dim=2)
        return {'logits': self.classifier(feature), 'gradcam_activation': gradcam_activation,
                'visualized_features': visualized_features}
