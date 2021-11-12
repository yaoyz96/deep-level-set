import torch.nn as nn
import torchvision.models.resnet as resnet
import torch
import numpy as np
from copy import deepcopy
import os
from torch.nn import functional as F
from mypath import Path
import torchvision.transforms as transforms
from .attention import (
    PAM_Module,
    CAM_Module,
    semanticModule,
    PAM_CAM_Layer,
    MultiConv)

affine_par = True

class Skip_Guided_Resnet(nn.Module):
    def __init__(self, concat_channels=128, mid_dim=256, final_dim=512, n_classes=(1 ,2), resnet=None, torch_model=False, use_conv=False):
        super(Skip_Guided_Resnet, self).__init__()

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

        # guided attention
        self.pam_attention_1_1= PAM_CAM_Layer(128, True)
        self.cam_attention_1_1= PAM_CAM_Layer(128, False)
        self.semanticModule_1_1 = semanticModule(128)

        if self.use_conv:
            conv_final_1 = nn.Conv2d( 4 *concat_channels, mid_dim, kernel_size=3, padding=1, stride=2,
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

        # guided attention
        attn_pam4 = self.pam_attention_1_1(conv1_f)  
        attn_cam4 = self.cam_attention_1_1(conv1_f)
        concat_features = torch.cat((attn_cam4, attn_pam4, layer1_f, layer2_f, layer4_f), dim=1)

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