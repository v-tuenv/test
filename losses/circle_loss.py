from math import gamma
from typing import ForwardRef
import torch 
from torch import nn as nn
from torch.nn import functional as F


class CircleLoss(nn.Module):
    def __init__(self, margin = 0.2, gamma = 128.,**kwargs):
        super().__init__()
        self.m = margin 
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
    def forward(self, inputs, label=None):
        # = log([1+\sum_i->k(\sum_j->L) ])
        # inputs size (batch, 2, 512)
        inputs = F.normalize(inputs, dim=-1) 
        
        similari_postive = F.linear(inputs[:,0,:], inputs[:,1,:]).view(-1)

        similari_negative = F.linear(inputs[:,0,:], inputs[:,0,:]).view(-1)

        ap = torch.clamp_min(-similari_postive.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(similari_negative.detach()+self.m, min=0.)

        delta_p  =1-self.m
        delta_n = self.m

        logit_p =  -ap*(similari_postive - delta_p) * self.gamma
        logit_n = an*(similari_negative-delta_n ) *self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss
        



