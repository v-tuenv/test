import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from metrics_module.accuracy import accuracy

class AAMSoftmax(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super(AAMSoftmax, self).__init__()

        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, embedding_dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.mmm = -1*math.sin(math.pi + self.m) * self.m
        self.thh = math.cos(math.pi + self.m)
        print('Initialised AAM-Softmax margin %.3f scale %.3f'%(self.m, self.s))

    def forward(self, x, label=None):
        assert len(x.shape) == 3
        label = label.repeat_interleave(x.shape[1])
        x = x.reshape(-1, self.in_feats)
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m #(cos(x+\phi)) = cos * cos - sin * sin
        # cos(theta - m)
        phi2 =cosine *(-1* self.cos_m) - sine * self.sin_m # cos(theta - m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)  # cos > cos(pi - margin) -> phi > pi -margin
            phi2 = torch.where((cosine - self.thh) > 0, phi2, cosine - self.mmm) 

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi2) + ((1.0 - one_hot) * phi)
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1