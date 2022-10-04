import torch
import torch.nn as nn
import model.modules as modules
from.cbam import resnet2cbam, attention_vis
from copy import deepcopy


class LocalGlobalDecomposition(nn.Module):
    def __init__(self, args, num_classes):
        super(LocalGlobalDecomposition, self).__init__()
        if 'resnet' not in args.backbone and 'resnext' not in args.backbone:
            raise NotImplementedError
        decomposed_layers = 1
        backbone = list(modules.Backbone(args).net.children())
        assert decomposed_layers < len(backbone)
        self.shared_feature_extractor = nn.Sequential(*backbone[:len(backbone) - decomposed_layers])
        self.global_feature_extractor = nn.Sequential(*deepcopy(backbone[len(backbone) - decomposed_layers:]))
        self.local_feature_extractor = nn.ModuleList([resnet2cbam(m, False, False)
                                                      for m in deepcopy(backbone[len(backbone) - decomposed_layers:])])
        self.args = args
        num_features = 512
        if args.backbone in ['resnet50', 'resnet101', 'resnet152']:
            num_features = 2048
        self.pooling = modules.GlobalAvgPooling()
        self.classifier = nn.Linear(num_features, num_classes)

        self.gradcam_module = dict()
        for module in list(self.shared_feature_extractor.modules())[::-1]:
            if isinstance(module, nn.Conv2d):
                self.gradcam_module['last_shared_conv'] = module
                break
        for module in list(self.global_feature_extractor.modules())[::-1]:
            if isinstance(module, nn.Conv2d):
                self.gradcam_module['last_global_conv'] = module
                break
        for module in list(self.local_feature_extractor.modules())[::-1]:
            if isinstance(module, nn.Conv2d):
                self.gradcam_module['last_local_conv'] = module
                break

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()
        shared_f = self.shared_feature_extractor(x)
        global_f = self.global_feature_extractor(shared_f)
        gradcam_activation['last_shared_conv'] = shared_f
        gradcam_activation['last_global_conv'] = global_f
        feature = shared_f
        for m in self.local_feature_extractor:
            feature = m(feature)
            attention = attention_vis(m)
            if attention is not None:
                visualized_features['attention_%d' % len(visualized_features)] = attention[:, 0, :, :]
        local_f = feature
        gradcam_activation['last_local_conv'] = local_f
        visualized_features['shared'] = (shared_f / (shared_f.norm(dim=(2, 3), keepdim=True) + 1e-12)).sum(dim=1)
        visualized_features['global'] = (global_f / (global_f.norm(dim=(2, 3), keepdim=True) + 1e-12)).sum(dim=1)
        visualized_features['local'] = (local_f / (local_f.norm(dim=(2, 3), keepdim=True) + 1e-8)).sum(dim=1)
        global_f = self.pooling(global_f)
        local_f = self.pooling(local_f)
        final_feature = torch.where(global_f > local_f, global_f, local_f)

        return {'logits': self.classifier(final_feature),
                'gradcam_activation': gradcam_activation, 'visualized_features': visualized_features}
