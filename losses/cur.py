'''
credit : https://github.com/HuangYG123/CurricularFace/blob/8b2f47318117995aa05490c05b455b113489917e/head/metrics.py#L70
'''
import torch
from torch import nn as nn
from torch.nn import functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from metrics_module.accuracy import accuracy
def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class CurricularFace(nn.Module):
    def __init__(self, embedding_dim, num_classes, scale = 30, margin = 0.3,**kwargs):
        super(CurricularFace, self).__init__()
            # embedding_dim, num_classes, margin=0.3, scale=15, easy_margin=False, use_sigmoid=True, **kwargs
        print('Using Curricular Face')

        self.in_features = embedding_dim
        self.out_features = num_classes
        self.m = margin
        self.s = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, embedding_dim), requires_grad=True)
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x, label):
        assert len(x.shape) == 3
        # label = label.repeat_interleave(x.shape[1])
        b,o,q = x.size()
        x = x.reshape(-1, self.in_features)
        if x.size(0) != label.size(0):
            label = label.repeat_interleave(o)
       
        
        # cos(theta)
        cos_theta = F.linear(F.normalize(x), F.normalize(self.weight))
       

     
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, x.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return nn.CrossEntropyLoss()(output,label), accuracy(output.detach(), label.detach(), topk=(1,))[0]