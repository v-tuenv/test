# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import re
from collections import defaultdict

from torch.utils.data import Dataset
import torch
import hashlib

def string2numeric_hash(text):
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

class BaseDataSet(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, label_list, path_list=None):
        self.label_list = label_list
        self.path_list = path_list
        self.label_index_dict = self._build_label_index_dict()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"


    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        pass

class HindexOnly(BaseDataSet):
    def __init__(self, label_list, path_list=None, wav_to_feature=None):
        super().__init__(label_list, path_list=path_list)
        self.wav_to_feature=wav_to_feature
    def __len__(self):
        return len(self.path_list)
    def __getitem__(self, index):
        
        path = self.path_list[index]
        
        label = self.label_list[index]

        feature,sr  = self.wav_to_feature(path)

        return torch.FloatTensor(feature), label

class HSIndex(HindexOnly):
    def __init__(self, label_list, path_list=None, wav_to_feature=None):
        super().__init__(label_list, path_list=path_list, wav_to_feature=wav_to_feature)
    def __len__(self):
        return len(self.path_list)
    def __getitem__(self, index):
        pathwav,pathaudio = self.path_list[index]
        label = self.label_list[index]
        feature_wav,sr = self.wav_to_feature(pathwav)
        feature_song,sr = self.wav_to_feature(pathaudio)
        return torch.FloatTensor(feature_wav), torch.FloatTensor(feature_song),label