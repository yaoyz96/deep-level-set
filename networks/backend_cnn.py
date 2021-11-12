import torch.nn as nn
import torchvision.models.resnet as resnet
import torch
import numpy as np
from copy import deepcopy
import os
from torch.nn import functional as F
from mypath import Path
import torchvision.transforms as transforms
import math

affine_par = True


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        out += residual
        out = self.relu(out)

        return out


class Selayer(nn.Module):

    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes / 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes / 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class SEResNet(nn.Module):

    def __init__(self, block, layers, n_classes, nInputChannels=3, classifier="atrous",
                 dilations=(2, 4), strides=(2, 2, 2, 1, 1), _print=False, feature_dim=2048):
        if _print:
            print("Constructing ResNet model...")
            print("Dilations: {}".format(dilations))
            print("Number of classes: {}".format(n_classes))
            print("Number of Input Channels: {}".format(nInputChannels))
        self.inplanes = 64
        self.classifier = classifier
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation__=dilations[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation__=dilations[1])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, n_classes)

        if classifier == "atrous":
            if _print:
                print('Initializing classifier: A-trous pyramid')
            self.layer5 = ClassifierModule([6, 12, 18, 24], [6, 12, 18, 24], n_classes)
        elif classifier == "psp":
            if _print:
                print('Initializing classifier: PSP')
            self.layer5 = PSPModule(in_features=feature_dim, out_features=feature_dim // 4, sizes=(1, 2, 3, 6),
                                    n_classes=n_classes)
        else:
            self.layer5 = None
        self.layer5_1 = None
        self.layer5_2 = None

        weight_init(self)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False

        layers = [block(self.inplanes, planes, stride, dilation_=dilation__, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.layer5_2 is not None:
            x0 = self.layer5(x)
            x1 = self.layer5_1(x)
            x2 = self.layer5_2(x)
            return x0, x1, x2
        if self.layer5_1 is not None:
            x0 = self.layer5(x)
            x1 = self.layer5_1(x)
            return x0, x1
        if self.layer5 is not None:
            x0 = self.layer5(x)
            return x0

        return x


class ClassifierModule(nn.Module):

    def __init__(self, dilation_series, padding_series, n_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, n_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module
    """
    def __init__(self, in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage_1(in_features, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_features * (len(sizes)//4 + 1), out_features)
        self.relu = nn.ReLU()
        self.final = nn.Conv2d(out_features, n_classes, kernel_size=1)

    def _make_stage_1(self, in_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, in_features//4, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(in_features//4, affine=affine_par)
        relu = nn.ReLU(inplace=True)

        for i in bn.parameters():
            i.requires_grad = False

        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_features, out_features):
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)

        for i in bn.parameters():
            i.requires_grad = False

        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        priors.append(feats)
        bottle = self.relu(self.bottleneck(torch.cat(priors, 1)))
        out = self.final(bottle)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes, nInputChannels=3, classifier="atrous",
                 dilations=(2, 4), strides=(2, 2, 2, 1, 1), _print=False, feature_dim=2048):
        if _print:
            print("Constructing ResNet model...")
            print("Dilations: {}".format(dilations))
            print("Number of classes: {}".format(n_classes))
            print("Number of Input Channels: {}".format(nInputChannels))
        self.classes = n_classes
        self.inplanes = 64
        self.classifier = classifier
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0]) # layers=[3,4,23,3]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation__=dilations[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation__=dilations[1])

        if classifier == "atrous":
            if _print:
                print('Initializing classifier: A-trous pyramid')
            self.layer5 = ClassifierModule([6, 12, 18, 24], [6, 12, 18, 24], n_classes)
        elif classifier == "psp":
            if _print:
                print('Initializing classifier: PSP')
            self.layer5 = PSPModule(in_features=feature_dim, out_features=feature_dim//4, sizes=(1, 2, 3, 6), n_classes=n_classes)
        else:
            self.layer5 = None
        self.layer5_1 = None
        self.layer5_2 = None

        weight_init(self)

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = [block(self.inplanes, planes, stride, dilation_=dilation__, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))

        return nn.Sequential(*layers)

    def forward(self, x, bbox=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # classifiers
        if self.layer5_3 is not None:
            x0 = self.layer5(x)
            x1 = self.layer5_1(x)
            x2 = self.layer5_2(x)
            x3 = self.layer5_3(x)
            return x0, x1, x2, x3
        if self.layer5_2 is not None:
            x0 = self.layer5(x)
            x1 = self.layer5_1(x)
            x2 = self.layer5_2(x)
            return x0, x1, x2
        if self.layer5_1 is not None:
            x0 = self.layer5(x)
            x1 = self.layer5_1(x)
            return x0, x1
        if self.layer5 is not None:
            x0 = self.layer5(x)
            return x0
        return x

    def load_pretrained_ms(self, base_network, nInputChannels=3):
        flag = 0
        for module, module_ori in zip(self.modules(), base_network.Scale.modules()):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if not flag and nInputChannels != 3:
                    module.weight[:, :3, :, :].data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                    for i in range(3, int(module.weight.data.shape[1])):
                        module.weight[:, i, :, :].data = deepcopy(module_ori.weight[:, -1, :, :][:, np.newaxis, :, :].data)
                    flag = 1
                elif module.weight.data.shape == module_ori.weight.data.shape:
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                else:
                    print('Skipping Conv layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
            elif isinstance(module, nn.BatchNorm2d) and isinstance(module_ori, nn.BatchNorm2d) \
                    and module.weight.data.shape == module_ori.weight.data.shape:
                module.weight.data = deepcopy(module_ori.weight.data)
                module.bias.data = deepcopy(module_ori.bias.data)

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = [self.layer5, self.layer5_1, self.layer5_2]
        for j in range(len(b)):
            if b[j] is not None:
                for k in b[j].parameters():
                    if k.requires_grad:
                        yield k


class MS_Deeplab(nn.Module):
    def __init__(self, block, NoLabels, nInputChannels=3):
        super(MS_Deeplab, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], NoLabels, nInputChannels=nInputChannels)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size=(int(input_size*0.75)+1, int(input_size*0.75)+1), mode='bilinear', align_corners=True)
        self.interp2 = nn.Upsample(size=(int(input_size*0.5)+1, int(input_size*0.5)+1), mode='bilinear', align_corners=True)
        self.interp3 = nn.Upsample(size=(outS(input_size), outS(input_size)), mode='bilinear', align_corners=True)
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.Scale(x))  # for original scale
        out.append(self.interp3(self.Scale(x2)))  # for 0.75x scale
        out.append(self.Scale(x3))  # for 0.5x scale

        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0], x2Out_interp)
        out.append(torch.max(temp1, x3Out_interp))
        return out[-1]


def Res_Deeplab(n_classes=21, pretrained=False):
    model = MS_Deeplab(Bottleneck, n_classes)
    if pretrained:
        pth_model = 'delse_epoch-49.pth'
        saved_state_dict = torch.load(os.path.join(Path.models_dir(), pth_model),
                                      map_location=lambda storage, loc: storage)
        if n_classes != 21:
            for i in saved_state_dict:
                i_parts = i.split('.')
                if i_parts[1] == 'layer5':
                    saved_state_dict[i] = model.state_dict()[i]
        model.load_state_dict(saved_state_dict)
    return model


class SkipResnet(nn.Module):
    def __init__(self, concat_channels=128, mid_dim=256, final_dim=512, n_classes=(1,2), resnet=None, torch_model=False, use_conv=False):
        super(SkipResnet, self).__init__()

        # Default transform for all torchvision models
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.concat_channels = concat_channels
        self.final_dim = final_dim

        assert resnet is not None
        self.resnet = resnet

        self.torch_model = torch_model
        self.use_conv = use_conv

        self.classes = n_classes

        concat1 = nn.Conv2d(64, concat_channels, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(concat_channels)
        relu1 = nn.ReLU(inplace=True)
        self.conv1_concat = nn.Sequential(concat1, bn1, relu1)

        concat2 = nn.Conv2d(256, concat_channels, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(concat_channels)
        relu2 = nn.ReLU(inplace=True)
        up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.res1_concat = nn.Sequential(concat2, bn2, relu2, up2)

        concat3 = nn.Conv2d(512, concat_channels, kernel_size=3, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(concat_channels)
        relu3 = nn.ReLU(inplace=True)
        up3 = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        self.res2_concat = nn.Sequential(concat3, bn3, relu3, up3)

        concat4 = nn.Conv2d(2048, concat_channels, kernel_size=3, padding=1, bias=False)
        bn4 = nn.BatchNorm2d(concat_channels)
        relu4 = nn.ReLU(inplace=True)
        up4 = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        self.res4_concat = nn.Sequential(concat4, bn4, relu4, up4)

        # Different from original, original used maxpool
        # Original used no activation here
        if self.use_conv:
            conv_final_1 = nn.Conv2d(4*concat_channels, mid_dim, kernel_size=3, padding=1, stride=2,
                bias=False)
            bn_final_1 = nn.BatchNorm2d(mid_dim)
            conv_final_2 = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, stride=2, bias=False)
            bn_final_2 = nn.BatchNorm2d(mid_dim)
            conv_final_3 = nn.Conv2d(mid_dim, final_dim, kernel_size=3, padding=1, bias=False)
            bn_final_3 = nn.BatchNorm2d(final_dim)

            self.conv_final = nn.Sequential(conv_final_1, bn_final_1, conv_final_2, bn_final_2,
                conv_final_3, bn_final_3)
        else:
            self.conv_final = None

    def forward(self, x):
        if self.torch_model:
            x = self.normalize(x)
        # Normalization

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        conv1_f = self.resnet.relu(x)
        x = self.resnet.maxpool(conv1_f)    # 112*112
        layer1_f = self.resnet.layer1(x)
        layer2_f = self.resnet.layer2(layer1_f)
        layer3_f = self.resnet.layer3(layer2_f)
        layer4_f = self.resnet.layer4(layer3_f)

        conv1_f = self.conv1_concat(conv1_f)   # [b,128,112,112]
        layer1_f = self.res1_concat(layer1_f)
        layer2_f = self.res2_concat(layer2_f)
        layer4_f = self.res4_concat(layer4_f)

        concat_features = torch.cat((conv1_f, layer1_f, layer2_f, layer4_f), dim=1)
        if self.use_conv:
            x = self.conv_final(concat_features)    # final feature map
        else:
            x = concat_features

        # classifiers
        if len(self.classes) >= 4:
            x0 = self.resnet.layer5(x)
            x1 = self.resnet.layer5_1(x)
            x2 = self.resnet.layer5_2(x)
            x3 = self.resnet.layer5_3(x)
            return x0, x1, x2, x3
        if self.resnet.layer5_2 is not None:
            x0 = self.resnet.layer5(x)
            x1 = self.resnet.layer5_1(x)
            x2 = self.resnet.layer5_2(x)
            return x0, x1, x2
        if self.resnet.layer5_1 is not None:
            x0 = self.resnet.layer5(x)
            x1 = self.resnet.layer5_1(x)
            return x0, x1
        if self.resnet.layer5 is not None:
            x0 = self.resnet.layer5(x)
            return x0
        return x

    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            out.append(self.normalizer(x))

        return torch.stack(out, dim=0)

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = [self.resnet.conv1, self.resnet.bn1, self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = [self.resnet.layer5, self.resnet.layer5_1, self.resnet.layer5_2,
             self.conv1_concat, self.res1_concat, self.res2_concat, self.res4_concat, self.conv_final]
        for j in range(len(b)):
            if b[j] is not None:
                for k in b[j].parameters():
                    if k.requires_grad:
                        yield k


def build_resnet_model(n_classes=(1, 2), pretrained=True, nInputChannels=4, classifier="atrous",
                       dilations=(2, 4), strides=(2, 2, 2, 1, 1), layers=(3, 4, 23, 3), feature_dim=2048):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, layers, n_classes[0], nInputChannels=nInputChannels,
                   classifier=classifier, dilations=dilations, strides=strides, _print=True, feature_dim=feature_dim)
    if pretrained:
        model_full = Res_Deeplab(n_classes[0], pretrained=pretrained)
        model.load_pretrained_ms(model_full, nInputChannels=nInputChannels)
    if len(n_classes) >= 2:
        model.layer5_1 = PSPModule(in_features=feature_dim, out_features=512, sizes=(1, 2, 3, 6), n_classes=n_classes[1])
        weight_init(model.layer5_1)
    if len(n_classes) >= 3:
        model.layer5_2 = PSPModule(in_features=feature_dim, out_features=512, sizes=(1, 2, 3, 6), n_classes=n_classes[2])
        weight_init(model.layer5_2)
    if len(n_classes) >= 4:
        model.layer5_3 = PSPModule(in_features=feature_dim, out_features=512, sizes=(1, 2, 3, 6), n_classes=n_classes[3])
        weight_init(model.layer5_3)
    return model


def backend_cnn_model(args):

    if args.backend_cnn == 'resnet101':
        model = build_resnet_model(n_classes=(1, 2, 1, 2), pretrained=False, nInputChannels=args.input_channels,
                                   classifier=args.classifier)
    elif args.backend_cnn == 'resnet101-skip4':
        concat_dim = args.concat_dim  # 128
        resnet = build_resnet_model(n_classes=(1, 2, 1, 2), pretrained=False, nInputChannels=args.input_channels,
                                    classifier=args.classifier, feature_dim=4*concat_dim)
        model = SkipResnet(concat_channels=concat_dim, resnet=resnet, n_classes=(1, 2, 1, 2))
    return model


def get_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4, model.layer5]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


