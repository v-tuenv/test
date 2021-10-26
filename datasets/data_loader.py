

import numpy as np

import torch
import wavfile
import math
import random
import glob,sys,os


import numpy as np
import random
import os
import glob
from argparse import Namespace
import yaml
from scipy import signal
from scipy.io import wavfile
from sklearn import metrics
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



def loadWAV(filename, max_frames, evalmode=True, num_eval=10):# SR = 16000 = 1s : 0.1s = 1 Frame -> 
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio = wavfile.read(filename)
    assert sample_rate == 16000
    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array(
            [np.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = np.stack(feats, axis=0).astype(np.float)

    return feat


def round_down(num, divisor):
    return num - (num % divisor)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class AugmentWAV(object):
    def __init__(self, musan_path, rir_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr = {
            'noise': [0, 15],
            'speech': [13, 20],
            'music': [5, 15]
        }
        self.numnoise = {'noise': [1, 1], 'speech': [1, 1], 'music': [1, 2]}
        self.noiselist = {}
        
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat],
                                  random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0],
                                       self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0]**2) + 1e-4)
            noises.append(
                np.sqrt(10**((clean_db - noise_db - noise_snr) / 10)) *
                noiseaudio)

        return np.sum(np.concatenate(noises, axis=0), axis=0,
                      keepdims=True) + audio

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)

        fs, rir = wavfile.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]


class VLSP_loader(Dataset):
    def __init__(self, dataset_file_name, augment, musan_path, rir_path,
                 max_frames, max_aug=1):
        self.max_aug=max_aug
        if augment is not True:
            self.max_aug=0
        self.augment_wav = AugmentWAV(musan_path=musan_path,
                                      rir_path=rir_path,
                                      max_frames=max_frames)

        self.dataset_file_name = dataset_file_name
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.augment = augment

        # Read Training Files...
        with open(dataset_file_name) as dataset_file:
            lines = dataset_file.readlines()

        dictkeys = list(set([x.split()[0] for x in lines]))
#         dictkeys.sort()
        dictkeys = {
            x:i for i,x in enumerate(dictkeys)
        }
#         dictkeys = SPK_ID
        self.label_dict = {}
        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            # filename = os.path.join(train_path, data[1])

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []

            self.label_dict[speaker_label].append(lidx)

            self.data_label.append(speaker_label)
            # self.data_list.append(filename)
            self.data_list.append(data[1].replace("data/train/competition_train","../input/d/tuenguyenasteam/vlsp-spkear-task/train/competition_train"))

    def __getitem__(self, indices):
        feat = []

        for index in indices:
            audio = loadWAV(self.data_list[index],
                            self.max_frames,
                            evalmode=False)

            if self.augment:
                augtype = random.randint(0, 4)
                if augtype == 1:
                    audio = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio = self.augment_wav.additive_noise('music', audio)
                elif augtype == 3:
                    audio = self.augment_wav.additive_noise('speech', audio) 
                elif augtype == 4:
                    audio = self.augment_wav.additive_noise('noise', audio)

            feat.append(audio)

        feat = np.concatenate(feat, axis=0)
#         print(indices)
        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

import numpy
import torch.distributed as dist
class VLSP_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

        self.data_label         = data_source.data_label;
        self.nPerSpeaker        = nPerSpeaker;
        self.max_seg_per_spk    = max_seg_per_spk;
        self.batch_size         = batch_size;
        self.epoch              = 0;
        self.seed               = seed;
        self.distributed        = distributed;
        self.weighted_label =[0,] * len(self.data_label)
    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = [];
            data_dict[speaker_label].append(index);


        ## Group file indices for each class
        dictkeys = list(data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []
        
        for findex, key in enumerate(dictkeys):
            
            data    = data_dict[key]
            weighted = [self.weighted_label[k] for k in data]
            new_data= [(x,i) for i,x in zip(data, weighted)]
            new_data.sort()
            
            data = [i[1] for i in new_data]
            
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)
            
            rp      = lol(numpy.arange(numSeg),self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid           = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size  = round_down(len(mixed_list), self.batch_size * dist.get_world_size()) 
            start_index = int ( ( dist.get_rank()     ) / dist.get_world_size() * total_size )
            end_index   = int ( ( dist.get_rank() + 1 ) / dist.get_world_size() * total_size )
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            for k in mixed_list[:total_size]:
                for kk in k:
                    self.weighted_label[kk] += 1
            return iter(mixed_list[:total_size])

    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter',
            torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size(
        )) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    fnr = fnr * 100
    fpr = fpr * 100

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) 

    return (tunedThreshold, eer, thresholds[idxE], metrics.auc(fpr, tpr))


def score_normalization(ref, com, cohorts, top=-1):
    """
    Adaptive symmetric score normalization using cohorts from eval data
    """
    def ZT_norm(ref, com, top=-1):
        """
        Perform Z-norm or T-norm depending on input order
        """
        S = np.mean(np.inner(cohorts, ref), axis=1)
        S = np.sort(S, axis=0)[::-1][:top]
        mean_S = np.mean(S)
        std_S = np.std(S)
        score = np.inner(ref, com)
        score = np.mean(score)
        return (score - mean_S) / std_S

    def S_norm(ref, com, top=-1):
        """
        Perform S-norm
        """
        return (ZT_norm(ref, com, top=top) + ZT_norm(com, ref, top=top)) / 2

    ref = ref.cpu().numpy()
    com = com.cpu().numpy()
    return S_norm(ref, com, top=top)


