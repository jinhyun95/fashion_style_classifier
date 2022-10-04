import torch
import torch.nn as nn
from copy import deepcopy

# FOLLOWING CODE IS AN IMPLEMENTATION OF [Residual Attention Network for Image Classification], WANG et al., CVPR 2017,
# CLONED FROM https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/blob/master/Residual-Attention-Network,
# WITH SOME ADJUSTMENTS FOR VISUALIZATION PURPOSES.

class AttentionModule_stage0(nn.Module):
    # input size is 112*112
    def __init__(self, in_channels, out_channels, args, size1=(112, 112), size2=(56, 56), size3=(28, 28), size4=(14, 14)):
        super(AttentionModule_stage0, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )
        if args.attentions is None or args.attentions == 0:
            self.attention = True
            self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 56*56
            self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
            self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
            self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 28*28
            self.softmax2_blocks = ResidualBlock(in_channels, out_channels)
            self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)
            self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 14*14
            self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
            self.skip3_connection_residual_block = ResidualBlock(in_channels, out_channels)
            self.mpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 7*7
            self.softmax4_blocks = nn.Sequential(
                ResidualBlock(in_channels, out_channels),
                ResidualBlock(in_channels, out_channels)
            )
            self.interpolation4 = nn.UpsamplingBilinear2d(size=size4)
            self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
            self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
            self.softmax6_blocks = ResidualBlock(in_channels, out_channels)
            self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
            self.softmax7_blocks = ResidualBlock(in_channels, out_channels)
            self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

            self.softmax8_blocks = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels , kernel_size=1, stride=1, bias = False),
                nn.Sigmoid()
            )
        else:
            self.attention = False
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        # 112*112
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        if self.attention:
            out_mpool1 = self.mpool1(x)
            # 56*56
            out_softmax1 = self.softmax1_blocks(out_mpool1)
            out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
            out_mpool2 = self.mpool2(out_softmax1)
            # 28*28
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
            out_mpool3 = self.mpool3(out_softmax2)
            # 14*14
            out_softmax3 = self.softmax3_blocks(out_mpool3)
            out_skip3_connection = self.skip3_connection_residual_block(out_softmax3)
            out_mpool4 = self.mpool4(out_softmax3)
            # 7*7
            out_softmax4 = self.softmax4_blocks(out_mpool4)
            out_interp4 = self.interpolation4(out_softmax4) + out_softmax3
            out = out_interp4 + out_skip3_connection
            out_softmax5 = self.softmax5_blocks(out)
            out_interp3 = self.interpolation3(out_softmax5) + out_softmax2
            out = out_interp3 + out_skip2_connection
            out_softmax6 = self.softmax6_blocks(out)
            out_interp2 = self.interpolation2(out_softmax6) + out_softmax1
            out = out_interp2 + out_skip1_connection
            out_softmax7 = self.softmax7_blocks(out)
            out_interp1 = self.interpolation1(out_softmax7) + out_trunk
            out_softmax8 = self.softmax8_blocks(out_interp1)
            out = (1 + out_softmax8) * out_trunk
        else:
            out = out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage1(nn.Module):
    # input size is 56*56
    def __init__(self, in_channels, out_channels, args, size1=(56, 56), size2=(28, 28), size3=(14, 14)):
        super(AttentionModule_stage1, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        if args.group is None:
            group = out_channels
        else:
            group = args.group
        assert out_channels % group == 0
        if args.attentions is None or args.attentions <= 1:
            self.attention = True
            self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
            self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
            self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.softmax2_blocks = ResidualBlock(in_channels, out_channels)
            self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)
            self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.softmax3_blocks = nn.Sequential(
                ResidualBlock(in_channels, out_channels),
                ResidualBlock(in_channels, out_channels)
            )
            self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
            self.softmax4_blocks = ResidualBlock(in_channels, out_channels)
            self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
            self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
            self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
            self.softmax6_blocks = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, group, kernel_size=1, stride=1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.attention = False
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        if self.attention:
            out_mpool1 = self.mpool1(x)
            out_softmax1 = self.softmax1_blocks(out_mpool1)
            out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
            out_mpool2 = self.mpool2(out_softmax1)
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
            out_mpool3 = self.mpool3(out_softmax2)
            out_softmax3 = self.softmax3_blocks(out_mpool3)
            out_interp3 = self.interpolation3(out_softmax3) + out_softmax2
            out = out_interp3 + out_skip2_connection
            out_softmax4 = self.softmax4_blocks(out)
            out_interp2 = self.interpolation2(out_softmax4) + out_softmax1
            out = out_interp2 + out_skip1_connection
            out_softmax5 = self.softmax5_blocks(out)
            out_interp1 = self.interpolation1(out_softmax5) + out_trunk
            out_softmax6 = self.softmax6_blocks(out_interp1)
            out = (1 + torch.cat([out_softmax6] * (out_trunk.size(1) // out_softmax6.size(1)), 1)) * out_trunk
        else:
            out = out_trunk
            out_softmax6 = torch.ones_like(out)
        out_last = self.last_blocks(out)

        return out_last, out_softmax6


class AttentionModule_stage2(nn.Module):
    # input image size is 28*28
    def __init__(self, in_channels, out_channels, args, size1=(28, 28), size2=(14, 14)):
        super(AttentionModule_stage2, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )
        if args.group is None:
            group = out_channels
        else:
            group = args.group
        assert out_channels % group == 0
        if args.attentions is None or args.attentions <= 2:
            self.attention = True
            self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
            self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
            self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.softmax2_blocks = nn.Sequential(
                ResidualBlock(in_channels, out_channels),
                ResidualBlock(in_channels, out_channels)
            )
            self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
            self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
            self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
            self.softmax4_blocks = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, group, kernel_size=1, stride=1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.attention = False
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        if self.attention:
            out_mpool1 = self.mpool1(x)
            out_softmax1 = self.softmax1_blocks(out_mpool1)
            out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
            out_mpool2 = self.mpool2(out_softmax1)
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            out_interp2 = self.interpolation2(out_softmax2) + out_softmax1
            out = out_interp2 + out_skip1_connection
            out_softmax3 = self.softmax3_blocks(out)
            out_interp1 = self.interpolation1(out_softmax3) + out_trunk
            out_softmax4 = self.softmax4_blocks(out_interp1)
            out = (1 + torch.cat([out_softmax4] * (out_trunk.size(1) // out_softmax4.size(1)), 1)) * out_trunk
        else:
            out = out_trunk
            out_softmax4 = torch.ones_like(out_trunk)
        out_last = self.last_blocks(out)

        return out_last, out_softmax4


class AttentionModule_stage3(nn.Module):
    # input image size is 14*14
    def __init__(self, in_channels, out_channels, args, size1=(14, 14)):
        super(AttentionModule_stage3, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )
        if args.group is None:
            group = out_channels
        else:
            group = args.group
        assert out_channels % group == 0
        if args.attentions is None or args.attentions <= 3:
            self.attention = True
            self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.softmax1_blocks = nn.Sequential(
                ResidualBlock(in_channels, out_channels),
                ResidualBlock(in_channels, out_channels)
            )
            self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
            self.softmax2_blocks = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, group, kernel_size=1, stride=1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.attention = False
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        if self.attention:
            out_mpool1 = self.mpool1(x)
            out_softmax1 = self.softmax1_blocks(out_mpool1)
            out_interp1 = self.interpolation1(out_softmax1) + out_trunk
            out_softmax2 = self.softmax2_blocks(out_interp1)
            out = (1 + torch.cat([out_softmax2] * (out_trunk.size(1) // out_softmax2.size(1)), 1)) * out_trunk
        else:
            out = out_trunk
            out_softmax2 = torch.ones_like(out_trunk)
        out_last = self.last_blocks(out)

        return out_last, out_softmax2


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels // 4, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels // 4, output_channels // 4, 3, stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels // 4, output_channels, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out


class ResidualAttentionModel(nn.Module):
    def __init__(self, args, num_classes):
        super(ResidualAttentionModel, self).__init__()
        if args.image_size == 448 and 'deep' in args.exp_name.lower():
            self.net = ResidualAttentionModel448Deep(args, num_classes)
        elif args.image_size == 448:
            self.net = ResidualAttentionModel448(args, num_classes)
        elif args.image_size == 224 and 'deep' in args.exp_name.lower():
            self.net = ResidualAttentionModel224Deep(args, num_classes)
        elif args.image_size == 224:
            self.net = ResidualAttentionModel224(args, num_classes)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.net(x)


class ResidualAttentionModel448Deep(nn.Module):
    def __init__(self, args, num_classes):
        super(ResidualAttentionModel448Deep, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block0 = ResidualBlock(64, 128)
        self.attention_module0 = AttentionModule_stage0(128, 128, args)
        self.residual_block1 = ResidualBlock(128, 256, 2)
        self.attention_module1 = AttentionModule_stage1(256, 256, args)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512, args)
        self.attention_module2_2 = AttentionModule_stage2(512, 512, args)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024, args)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024, args)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024, args)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048, num_classes)
        self.gradcam_module = dict()
        self.gradcam_module['last_conv'] = self.residual_block6.conv3
        self.gradcam_module['att_1_1'] = list(self.attention_module1.softmax6_blocks.children())[-2]
        self.gradcam_module['att_2_1'] = list(self.attention_module2.softmax4_blocks.children())[-2]
        self.gradcam_module['att_2_2'] = list(self.attention_module2_2.softmax4_blocks.children())[-2]
        self.gradcam_module['att_3_1'] = list(self.attention_module3.softmax2_blocks.children())[-2]
        self.gradcam_module['att_3_2'] = list(self.attention_module3_2.softmax2_blocks.children())[-2]
        self.gradcam_module['att_3_3'] = list(self.attention_module3_3.softmax2_blocks.children())[-2]

    def forward(self, x):
        gradcam_activation = dict()
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block0(out)
        out = self.attention_module0(out)
        out = self.residual_block1(out)
        out, att_1_1 = self.attention_module1(out)
        out = self.residual_block2(out)
        out, att_2_1 = self.attention_module2(out)
        out, att_2_2 = self.attention_module2_2(out)
        out = self.residual_block3(out)
        out, att_3_1 = self.attention_module3(out)
        out, att_3_2 = self.attention_module3_2(out)
        out, att_3_3 = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        gradcam_activation['last_conv'] = deepcopy(out.detach())
        gradcam_activation['att_1_1'] = deepcopy(att_1_1.detach())
        gradcam_activation['att_2_1'] = deepcopy(att_2_1.detach())
        gradcam_activation['att_2_2'] = deepcopy(att_2_2.detach())
        gradcam_activation['att_3_1'] = deepcopy(att_3_1.detach())
        gradcam_activation['att_3_2'] = deepcopy(att_3_2.detach())
        gradcam_activation['att_3_3'] = deepcopy(att_3_3.detach())
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {'logits': out, 'gradcam_activation': gradcam_activation, 'visualized_features': dict()}


class ResidualAttentionModel448(nn.Module):
    def __init__(self, args, num_classes):
        super(ResidualAttentionModel448, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block0 = ResidualBlock(64, 128)
        self.attention_module0 = AttentionModule_stage0(128, 128, args)
        self.residual_block1 = ResidualBlock(128, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256, args)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512, args)
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024, args)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,num_classes)
        self.gradcam_module = dict()
        self.gradcam_module['last_conv'] = self.residual_block6.conv3
        self.gradcam_module['att_1'] = list(self.attention_module1.softmax6_blocks.children())[-2]
        self.gradcam_module['att_2'] = list(self.attention_module2.softmax4_blocks.children())[-2]
        self.gradcam_module['att_3'] = list(self.attention_module3.softmax2_blocks.children())[-2]

    def forward(self, x):
        gradcam_activation = dict()
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block0(out)
        out = self.attention_module0(out)
        out = self.mpool1(out)
        out = self.residual_block1(out)
        out, att1 = self.attention_module1(out)
        out = self.residual_block2(out)
        out, att2 = self.attention_module2(out)
        out = self.residual_block3(out)
        out, att3 = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        gradcam_activation['last_conv'] = deepcopy(out.detach())
        gradcam_activation['att_1'] = deepcopy(att1.detach())
        gradcam_activation['att_2'] = deepcopy(att2.detach())
        gradcam_activation['att_3'] = deepcopy(att3.detach())
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {'logits': out, 'gradcam_activation': gradcam_activation, 'visualized_features': dict()}


class ResidualAttentionModel224Deep(nn.Module):
    def __init__(self, args, num_classes):
        super(ResidualAttentionModel224Deep, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256, args)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512, args)
        self.attention_module2_2 = AttentionModule_stage2(512, 512, args)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024, args)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024, args)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024, args)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,num_classes)
        self.gradcam_module = dict()
        self.gradcam_module['last_conv'] = self.residual_block6.conv3
        self.gradcam_module['att_1_1'] = list(self.attention_module1.softmax6_blocks.children())[-2]
        self.gradcam_module['att_2_1'] = list(self.attention_module2.softmax4_blocks.children())[-2]
        self.gradcam_module['att_2_2'] = list(self.attention_module2_2.softmax4_blocks.children())[-2]
        self.gradcam_module['att_3_1'] = list(self.attention_module3.softmax2_blocks.children())[-2]
        self.gradcam_module['att_3_2'] = list(self.attention_module3_2.softmax2_blocks.children())[-2]
        self.gradcam_module['att_3_3'] = list(self.attention_module3_3.softmax2_blocks.children())[-2]

    def forward(self, x):
        gradcam_activation = dict()
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block1(out)
        out, att_1_1 = self.attention_module1(out)
        out = self.residual_block2(out)
        out, att_2_1 = self.attention_module2(out)
        out, att_2_2 = self.attention_module2_2(out)
        out = self.residual_block3(out)
        out, att_3_1 = self.attention_module3(out)
        out, att_3_2 = self.attention_module3_2(out)
        out, att_3_3 = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        gradcam_activation['last_conv'] = deepcopy(out.detach())
        gradcam_activation['att_1_1'] = deepcopy(att_1_1.detach())
        gradcam_activation['att_2_1'] = deepcopy(att_2_1.detach())
        gradcam_activation['att_2_2'] = deepcopy(att_2_2.detach())
        gradcam_activation['att_3_1'] = deepcopy(att_3_1.detach())
        gradcam_activation['att_3_2'] = deepcopy(att_3_2.detach())
        gradcam_activation['att_3_3'] = deepcopy(att_3_3.detach())
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {'logits': out, 'gradcam_activation': gradcam_activation, 'visualized_features': dict()}


class ResidualAttentionModel224(nn.Module):
    def __init__(self, args, num_classes):
        super(ResidualAttentionModel224, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256, args)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512, args)
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024, args)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,num_classes)
        self.gradcam_module = dict()
        self.gradcam_module['last_conv'] = self.residual_block6.conv3
        self.gradcam_module['att_1'] = list(self.attention_module1.softmax6_blocks.children())[-2]
        self.gradcam_module['att_2'] = list(self.attention_module2.softmax4_blocks.children())[-2]
        self.gradcam_module['att_3'] = list(self.attention_module3.softmax2_blocks.children())[-2]

    def forward(self, x):
        gradcam_activation = dict()
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block1(out)
        out, att1 = self.attention_module1(out)
        out = self.residual_block2(out)
        out, att2 = self.attention_module2(out)
        out = self.residual_block3(out)
        out, att3 = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        gradcam_activation['last_conv'] = deepcopy(out.detach())
        gradcam_activation['att_1'] = deepcopy(att1.detach())
        gradcam_activation['att_2'] = deepcopy(att2.detach())
        gradcam_activation['att_3'] = deepcopy(att3.detach())
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {'logits': out, 'gradcam_activation': gradcam_activation, 'visualized_features': dict()}
