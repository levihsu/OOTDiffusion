#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   AugmentCE2P.py
@Time    :   8/4/19 3:35 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import functools
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
# Note here we adopt the InplaceABNSync implementation from https://github.com/mapillary/inplace_abn
# By default, the InplaceABNSync module contains a BatchNorm Layer and a LeakyReLu layer
from modules import InPlaceABNSync
import numpy as np

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

affine_par = True

pretrained_settings = {
    'resnet101': {
        'imagenet': {
            'input_space': 'BGR',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.406, 0.456, 0.485],
            'std': [0.225, 0.224, 0.229],
            'num_classes': 1000
        }
    },
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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

        out = out + residual
        out = self.relu_inplace(out)

        return out


class CostomAdaptiveAvgPool2D(nn.Module):

    def __init__(self, output_size):

        super(CostomAdaptiveAvgPool2D, self).__init__()

        self.output_size = output_size

    def forward(self, x):

        H_in, W_in = x.shape[-2:]
        H_out, W_out = self.output_size

        out_i = []
        for i in range(H_out):
            out_j = []
            for j in range(W_out):
                hs = int(np.floor(i * H_in / H_out))
                he = int(np.ceil((i + 1) * H_in / H_out))

                ws = int(np.floor(j * W_in / W_out))
                we = int(np.ceil((j + 1) * W_in / W_out))

                # print(hs, he, ws, we)
                kernel_size = [he - hs, we - ws]

                out = F.avg_pool2d(x[:, :, hs:he, ws:we], kernel_size)
                out_j.append(out)

            out_j = torch.concat(out_j, -1)
            out_i.append(out_j)

        out_i = torch.concat(out_i, -2)
        return out_i


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        tmp = []
        for size in sizes:
            if size == 3 or size == 6:
                tmp.append(self._make_stage_custom(features, out_features, size))
            else:
                tmp.append(self._make_stage(features, out_features, size))
        self.stages = nn.ModuleList(tmp)
        # self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            InPlaceABNSync(out_features),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def _make_stage_custom(self, features, out_features, size):
        prior = CostomAdaptiveAvgPool2D(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            InPlaceABNSync(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


class Edge_Module(nn.Module):
    """
    Edge Learning Branch
    """

    def __init__(self, in_fea=[256, 512, 1024], mid_fea=256, out_fea=2):
        super(Edge_Module, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea


class Decoder_Module(nn.Module):
    """
    Parsing Branch Decoder Module.
    """

    def __init__(self, num_classes):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(x, x2)
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fushion(x)
        return [[parsing_result, fusion_result], edge_result]


def initialize_pretrained_model(model, settings, pretrained='./models/resnet101-imagenet.pth'):
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

    if pretrained is not None:
        saved_state_dict = torch.load(pretrained)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
        model.load_state_dict(new_params)


def resnet101(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model
