import torch
import torch.nn as nn
import model.modules as modules


class BaselineClassifier(nn.Module):
    def __init__(self, args, num_classes):
        super(BaselineClassifier, self).__init__()
        self.feature_extractor = modules.Backbone(args)
        self.args = args
        if args.backbone in ['resnet18', 'resnet34']:
            num_features = 512
        elif 'mobilenet' in args.backbone:
            num_features = 1280
        else:
            num_features = 2048
        if 'resnet' in args.backbone or 'resnext' in args.backbone:
            if args.baseline_pooltype == 'avg':
                self.classifier = nn.Sequential(modules.GlobalAvgPooling(),
                                                nn.Linear(num_features, num_classes))
            else:
                self.classifier = nn.Sequential(modules.GlobalMaxPooling(),
                                                nn.Linear(num_features, num_classes))
        elif 'mobilenet' in args.backbone:
            self.classifier = nn.Sequential(
                modules.GlobalAvgPooling(),
                nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(1280, num_classes)
                )
            )
        else:
            raise NotImplementedError
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        visualized_features = dict()
        feature = self.feature_extractor(x)
        if self.args.mode == 'visualize':
            feature.register_hook(self.save_gradient)
        # visualized_features['last_conv'] = (feature / (feature.norm(dim=(2, 3), keepdim=True) + 1e-12)).sum(dim=1)
        logits = self.classifier(feature)
        return {'logits': logits, 'gradcam_activation': feature,
                'visualized_features': visualized_features}
