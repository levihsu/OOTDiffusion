#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   soft_dice_loss.py
@Time    :   8/13/19 5:09 PM
@Desc    :   
@License :   This source code is licensed under the license found in the 
             LICENSE file in the root directory of this source tree.
"""

from __future__ import print_function, division

import torch
import torch.nn.functional as F
from torch import nn

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def tversky_loss(probas, labels, alpha=0.5, beta=0.5, epsilon=1e-6):
    '''
    Tversky loss function.
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)

    Same as soft dice loss when alpha=beta=0.5.
    Same as Jaccord loss when alpha=beta=1.0.
    See `Tversky loss function for image segmentation using 3D fully convolutional deep networks`
    https://arxiv.org/pdf/1706.05721.pdf
    '''
    C = probas.size(1)
    losses = []
    for c in list(range(C)):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        p0 = class_pred
        p1 = 1 - class_pred
        g0 = fg
        g1 = 1 - fg
        numerator = torch.sum(p0 * g0)
        denominator = numerator + alpha * torch.sum(p0 * g1) + beta * torch.sum(p1 * g0)
        losses.append(1 - ((numerator) / (denominator + epsilon)))
    return mean(losses)


def flatten_probas(probas, labels, ignore=255):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class SoftDiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(SoftDiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        pred = F.softmax(pred, dim=1)
        return tversky_loss(*flatten_probas(pred, label, ignore=self.ignore_index), alpha=0.5, beta=0.5)


class SoftJaccordLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(SoftJaccordLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        pred = F.softmax(pred, dim=1)
        return tversky_loss(*flatten_probas(pred, label, ignore=self.ignore_index), alpha=1.0, beta=1.0)
