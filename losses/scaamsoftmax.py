from torch import nn as nn
from torch.nn import functional as F
import torch
import math
from metrics_module.accuracy import accuracy

class SCAAMsoftmax(nn.Module):
    def __init__(self,embedding_dim, num_classes,margin=0.3, scale = 30., easy_margin=False,k_center=3, **kwargs):
        super().__init__()
        self.test_normalize = True
        self.out_features = num_classes
        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.k = k_center
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes * k_center, embedding_dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        
        print("SCAAMsoftmax")
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x, label=None):
#         x=inputs[:,0,:]
        x=x.squeeze(1)
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats,x.size()
#         nlossP, _       = self.angleproto(inputs,None)
        # cos(theta)
        cosine_all = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
      
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        
        
        return loss , prec1