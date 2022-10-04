import torch
import torch.nn as nn
import model.modules as modules
from torchvision.models.resnet import BasicBlock, Bottleneck


def attention_calculation(gaussian_params, feature_size):
    m_x = gaussian_params[:, :, 0].sigmoid() * float(feature_size[0])
    m_y = gaussian_params[:, :, 1].sigmoid() * float(feature_size[1])
    # TODO: SIGMA SCALING
    s_x = gaussian_params[:, :, 2].sigmoid() * float(feature_size[0])
    s_y = gaussian_params[:, :, 3].sigmoid() * float(feature_size[1])
    rho = gaussian_params[:, :, 4].sigmoid()
    attention = []
    for i in range(feature_size):
        y_axis_attention = []
        for j in range(feature_size):
            y_axis_attention.append(
                (- 0.5 * (1 - rho.pow(2)).pow(-1) * ((m_x - i).pow(2).div(s_x.pow(2)) + (m_y - j).pow(2).div(s_y.pow(2))
                                                     - 2 * rho * (m_x - i) * (m_y - j).div(s_x * s_y))).exp())
        attention.append(torch.stack(y_axis_attention, -1))
    attention = torch.stack(attention, dim=2)
    attention /= attention.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
    return attention


class GaussianAttentionModule(nn.Module):
    def __init__(self, group, in_feature):
        super(GaussianAttentionModule, self).__init__()
        self.group = group
        self.conv = nn.Sequential(
            nn.Conv2d(in_feature, in_feature, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=in_feature // 2, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)))
        self.fc = nn.ModuleList([nn.Linear(3 * 3 * in_feature // group, 5, bias=False)
                                 for _ in range(group)])
        self.bn_post = nn.ModuleList([nn.BatchNorm1d(num_features=5, eps=1e-5, momentum=0.01, affine=True)
                                      for _ in range(group)])
        for key in self.bn_post.state_dict().keys():
            self.bn_post.state_dict()[key][...] = 0.

    def forward(self, x):
        x = self.relu(self.bn_pre(self.conv(x)))
        groups = torch.split(x, x.size(1) // self.group, dim=1)
        gaussian_params = []
        for g in range(self.group):
            avg_pool = groups[g].mean(1)
            max_pool = groups[g].max(1)[0]
            pooled_feature = torch.cat([torch.flatten(avg_pool, start_dim=1), torch.flatten(max_pool, start_dim=1)], 1)
            gaussian_params.append(self.bn_post[g](self.fc[g](pooled_feature)))
        gaussian_params = torch.stack(gaussian_params, 1)

        return gaussian_params


class BasicBlockGA(BasicBlock):
    def __init__(self, inplanes, planes, group, stride=1, downsample=None):
        super(BasicBlockGA, self).__init__(inplanes, planes, stride, downsample)
        self.attention_module = GaussianAttentionModule(group, self.conv3.out_channels)
        self.attention_heatmap = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        gaussian_parameters = self.attention_module(out)
        self.attention_heatmap = attention_calculation(gaussian_parameters, out.size()[2:])
        out = out * torch.cat([self.attention_heatmap] * self.attention_module.group, 1)

        out += residual
        out = self.relu(out)

        return out


class BottleneckGA(Bottleneck):
    def __init__(self, inplanes, planes, group, stride=1, downsample=None):
        super(BottleneckGA, self).__init__(inplanes, planes, stride, downsample)
        self.attention_module = GaussianAttentionModule(group, self.conv3.out_channels)
        self.attention_heatmap = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        gaussian_parameters = self.attention_module(out)
        self.attention_heatmap = attention_calculation(gaussian_parameters, out.size()[2:])
        out = out * torch.cat([self.attention_heatmap] * self.attention_module.group, 1)

        out += residual
        out = self.relu(out)

        return out


def resblock_gaussian_insertion(resblock, group):
    if isinstance(resblock, BasicBlock):
        block = BasicBlockGA(resblock.conv1.in_channels,
                             resblock.conv3.out_channels // resblock.expansion,
                             group,
                             resblock.conv2.stride, resblock.downsample)
        state_dict = resblock.state_dict()
        block.load_state_dict(state_dict, strict=False)
        return block

    elif isinstance(resblock, Bottleneck):
        block = BottleneckGA(resblock.conv1.in_channels,
                             resblock.conv3.out_channels // resblock.expansion,
                             group,
                             resblock.conv2.stride, resblock.downsample)
        state_dict = resblock.state_dict()
        block.load_state_dict(state_dict, strict=False)
        return block

    elif isinstance(resblock, nn.Sequential):
        return nn.Sequential(*[resblock_gaussian_insertion(m, group) for m in resblock.children()])

    else:
        return resblock


def attention_vis(module):
    if isinstance(module, BasicBlockGA) or isinstance(module, BottleneckGA):
        return module.attention_heatmap
    elif isinstance(module, nn.Sequential):
        for m in list(module.children())[::-1]:
            attention = attention_vis(m)
            if attention is not None:
                return attention
        return None
    return None

class GaussianAttentionResNet(nn.Module):
    def __init__(self, args, num_classes):
        super(GaussianAttentionResNet, self).__init__()
        if 'resnet' not in args.backbone and 'resnext' not in args.backbone:
            raise NotImplementedError
        modulelist = [resblock_gaussian_insertion(m, args.attentions) for m in modules.Backbone(args).net.children()]
        self.feature_extractor = nn.ModuleList(modulelist)
        num_features = 512
        if args.backbone in ['resnet50', 'resnet101', 'resnet152']:
            num_features = 2048
        self.classifier = nn.Sequential(modules.GlobalAvgPooling(), nn.Linear(num_features, num_classes))

        for m in list(self.feature_extractor[-1].modules())[::-1]:
            if isinstance(m, BasicBlockGA) or isinstance(m, BottleneckGA):
                self.gradcam_module = {'last_conv': m.conv3}
                break

    def forward(self, x):
        gradcam_activation = dict()
        visualized_features = dict()
        layer_ix = 1
        for m in self.feature_extractor:
            x = m(x)
            attention = attention_vis(m)
            if attention is not None:
                for head_ix in range(attention.size(1)):
                    visualized_features['att_layer%d_head%02d' % (layer_ix, head_ix)] = attention[:, head_ix, :, :]
                layer_ix += 1
        gradcam_activation['last_conv'] = x
        logits = self.classifier(x)
        return {'logits': logits, 'gradcam_activation': gradcam_activation, 'visualized_features': visualized_features}
