#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   kl_loss.py
@Time    :   7/23/19 4:02 PM
@Desc    :   
@License :   This source code is licensed under the license found in the 
             LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn.functional as F
from torch import nn
from datasets.target_generation import generate_edge_tensor


class ConsistencyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(ConsistencyLoss, self).__init__()
        self.ignore_index=ignore_index

    def forward(self, parsing, edge, label):
        parsing_pre = torch.argmax(parsing, dim=1)
        parsing_pre[label==self.ignore_index]=self.ignore_index
        generated_edge = generate_edge_tensor(parsing_pre)
        edge_pre = torch.argmax(edge, dim=1)
        v_generate_edge = generated_edge[label!=255]
        v_edge_pre = edge_pre[label!=255]
        v_edge_pre = v_edge_pre.type(torch.cuda.FloatTensor)
        positive_union = (v_generate_edge==1)&(v_edge_pre==1) # only the positive values count
        return F.smooth_l1_loss(v_generate_edge[positive_union].squeeze(0), v_edge_pre[positive_union].squeeze(0))
