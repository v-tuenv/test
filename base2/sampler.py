import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - dataset (BaseDataSet).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, dataset, batch_size, num_instances, max_iters):
        self.label_index_dict = dataset.label_index_dict
        self.batch_size = batch_size
        self.K = num_instances
        self.num_labels_per_batch = self.batch_size // self.K
        self.max_iters = max_iters
        self.labels = list(self.label_index_dict.keys())

    def __len__(self):
        return self.max_iters

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"|Sampler| iters {self.max_iters}| K {self.K}| M {self.batch_size}|"

    def _prepare_batch(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.label_index_dict[label])
            if len(idxs) < self.K:
                idxs.extend(np.random.choice(idxs, size=self.K - len(idxs), replace=True))
            random.shuffle(idxs)

            batch_idxs_dict[label] = [idxs[i * self.K: (i + 1) * self.K] for i in range(len(idxs) // self.K)]

        avai_labels = copy.deepcopy(self.labels)
        return batch_idxs_dict, avai_labels

    def __iter__(self):
        batch_idxs_dict, avai_labels = self._prepare_batch()
        for _ in range(self.max_iters):
            batch = []
            if len(avai_labels) < self.num_labels_per_batch:
                batch_idxs_dict, avai_labels = self._prepare_batch()

            selected_labels = random.sample(avai_labels, self.num_labels_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                batch.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)
            yield batch



import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import numpy
def round_down(num, divisor):
    return num - (num%divisor)
class voxceleb_sampler(Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size):

        self.label_dict         = data_source.label_index_dict
        self.nPerSpeaker        = nPerSpeaker
        self.max_seg_per_spk    = max_seg_per_spk
        self.batch_size         = batch_size
        self.data_source = data_source
        
    def __iter__(self):
        
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data    = self.label_dict[key]
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)
            
            rp      = lol(numpy.random.permutation(len(data))[:numSeg],self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Data in random order
        mixid           = numpy.random.permutation(len(flattened_label))
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)
        
        return iter([flattened_list[i] for i in mixmap])
    
    def __len__(self):
        return len(self.data_source)