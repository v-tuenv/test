from torch import nn as nn
from losses.aamsoftmax import AAMSoftmax
import torch
from metrics_module.accuracy import accuracy
import numpy
from torch.nn import functional as F
from losses.softmax_face import MarginSoftmaxLoss
class angleproto(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(angleproto, self).__init__()

        self.test_normalize = True
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label   = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        nloss   = self.criterion(cos_sim_matrix, label)
        prec1   = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1
class softmaxprototypical(nn.Module):

    def __init__(self, **kwargs):
        super(softmaxprototypical, self).__init__()

        self.test_normalize = True
        dtype = kwargs.get("final_layer","MarginSoftmaxLoss")
        if dtype == 'AAM':
            self.softmax = AAMSoftmax( 
                embedding_dim = kwargs.get("embedding_dim"), num_classes=kwargs.get("num_classes"), margin=0.2, scale=30.,
            )
        else:
            self.softmax = MarginSoftmaxLoss(
                input_dim=kwargs.get("embedding_dim"),
                num_targets = kwargs.get("num_classes"),
                ring_loss=0.2,
                inter_loss=0.1
            )
            print("Init marginLoss ")
        self.angleproto = angleproto(**kwargs)

        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1   = self.softmax(x.reshape(-1,1,x.size()[-1]), label.repeat_interleave(2))

        nlossP, _       = self.angleproto(x,None)

        return nlossS+nlossP, prec1