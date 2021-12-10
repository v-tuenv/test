import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base2.mfcc import MfccSpecAug

''' Res2Conv1d + BatchNorm1d + ReLU
'''


class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width, momentum=0.8))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class ResNetBasic(nn.Module):
    '''
    in_channels == out_channels == channels
    '''

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1,
                 bias=False, scale=None):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                                stride=stride, dilation=dilation, bias=bias)
        self.bn_1 = nn.BatchNorm1d(channels)

        self.conv_2 = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                                stride=stride, dilation=dilation, bias=bias)
        self.bn_2 = nn.BatchNorm1d(channels)

    def forward(self, x) -> torch.Tensor:
        residual = x
        out = self.bn_1(F.relu(self.conv_1(x)))
        out = self.bn_2(self.conv_2(out))
        out = F.relu(out + residual)
        return out


''' Conv1d + BatchNorm1d + ReLU
'''


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
        )
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.8)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SqueezeExcitation(nn.Module):
    ''' The SE connection of 1D case. '''
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


''' SE-Res2Block.
    Note: residual connection is implemented in the ECAPA_TDNN model, not here.
'''


class SE_Res2Block(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.net = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            SqueezeExcitation(channels)
        )
        self.skip_layer = Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0)
        self.residual_layer = Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.net(x)
        res = (self.residual_layer(h) + x)*math.sqrt(0.5)
        skip = self.skip_layer(h)
        return res, skip


''' Attentive weighted mean and standard deviation pooling.
'''


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
        )  # equals W and b in the paper
        self.linear2 = nn.Sequential(
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
        )
        # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class PreEmphasis(torch.nn.Module):
    """ Adapt from https://github.com/clovaai/voxceleb_trainer/blob/master/utils.py """
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert len(x.shape) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        x = x.unsqueeze(1)
        x = F.pad(x, [1, 0], 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)


class EcapaTDNN(nn.Module):
    def __init__(self, n_features=80, n_channels=512,
                 train_augment=False, feature_type="mfcc", loss_type="aam",
                 emb_dim=192, n_classes=400, m=0.2, s=30):
        super().__init__()
        self.pre_emp = PreEmphasis()
        self.train_augment = train_augment
        assert feature_type in ["mfcc"], "Un-supported feature type %s" % feature_type
        self.feature_layer = MfccSpecAug(n_mfcc=n_features, sample_rate=16000,
                                        melkwargs={"n_fft": 512, "win_length": 400,
                                                   "hop_length": 160,
                                                   "window_fn": torch.hamming_window,
                                                   "n_mels": n_features}
                                         )
        self.mfcc_norm = nn.InstanceNorm1d(n_features)
        self.layer1 = Conv1dReluBn(n_features, n_channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(n_channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(n_channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(n_channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)
        self.layer5 = SE_Res2Block(n_channels, kernel_size=3, stride=1, padding=5, dilation=5, scale=8)
        cat_channels = n_channels * 4
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        self.pooling = AttentiveStatsPool(cat_channels, 128)
        self.bn1 = nn.BatchNorm1d(cat_channels * 2, momentum=0.25)
        self.dropout = nn.Dropout(p = 0.3)
        self.linear = nn.Linear(cat_channels * 2, emb_dim)
        self.bn2 = nn.BatchNorm1d(emb_dim, momentum=0.25)

    def forward(self, x):
        """ Forward pass """
        mfcc = self.mfcc_norm(self.feature_layer(self.pre_emp(x), augment=self.train_augment))
        out1 = self.layer1(mfcc)
        out2, skip2 = self.layer2(out1)
        out3, skip3 = self.layer3(out2)
        out4, skip4 = self.layer4(out3)
        out5, skip5 = self.layer5(out4)

        out = torch.cat([skip2, skip3, skip4, skip5], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.dropout(out)
        out = self.bn2(self.linear(out))
        return out