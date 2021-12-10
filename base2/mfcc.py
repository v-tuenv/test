import torch
from torch import Tensor
from torchaudio import functional as F
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
import math
from typing import Callable, Optional
from warnings import warn
import numpy as np


class MfccSpecAug(torch.nn.Module):
    r""" Custom MFCC layer for SpecAug
    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_mfcc (int, optional): Number of mfc coefficients to retain. (Default: ``40``)
        dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: ``2``)
        norm (str, optional): norm to use. (Default: ``'ortho'``)
        log_mels (bool, optional): whether to use log-mel spectrograms instead of db-scaled. (Default: ``False``)
        melkwargs (dict or None, optional): arguments for MelSpectrogram. (Default: ``None``)
    """
    __constants__ = ['sample_rate', 'n_mfcc', 'dct_type', 'top_db', 'log_mels']

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 40,
                 dct_type: int = 2,
                 norm: str = 'ortho',
                 log_mels: bool = False,
                 melkwargs: Optional[dict] = None) -> None:
        super(MfccSpecAug, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported'.format(dct_type))
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = AmplitudeToDB('power', self.top_db)
        if melkwargs is not None:
            self.MelSpectrogram = MelSpectrogram(sample_rate=self.sample_rate, **melkwargs)
        else:
            self.MelSpectrogram = MelSpectrogram(sample_rate=self.sample_rate)

        if self.n_mfcc > self.MelSpectrogram.n_mels:
            raise ValueError('Cannot select more MFCC coefficients than # mel bins')
        dct_mat = F.create_dct(self.n_mfcc, self.MelSpectrogram.n_mels, self.norm).contiguous()
        self.register_buffer('dct_mat', dct_mat)
        self.log_mels = log_mels

    def forward(self, waveform: Tensor, augment=False) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).
            augment (bool): enable SpecAug
        Returns:
            Tensor: specgram_mel_db of size (..., ``n_mfcc``, time).
        """

        # pack batch
        shape = waveform.size()
        waveform = waveform.view(-1, shape[-1])
        mel_specgram = self.MelSpectrogram(waveform)
        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)
        if augment:
            mel_specgram = self.spectrum_masking(mel_specgram)
        # (channel, n_mels, time).tranpose(...) dot (n_mels, n_mfcc)
        # -> (channel, time, n_mfcc).tranpose(...)
        mfcc = torch.matmul(mel_specgram.transpose(1, 2), self.dct_mat).transpose(1, 2)

        # unpack batch
        mfcc = mfcc.view(shape[:-1] + mfcc.shape[-2:])

        return mfcc

    def spectrum_masking(self, mel,
                         max_n_masked_band=8,
                         max_n_masked_frame=5):
        assert len(mel.shape) == 3
        assert mel.shape[1] == self.MelSpectrogram.n_mels
        n_channel = mel.shape[1]
        n_frame = mel.shape[2]
        for i, _ in enumerate(mel):
            # fill minimum value of spectrogram
            fill_value = torch.min(mel[i])
            for j in range(np.random.randint(1, 4)):
                n_masked_band = np.random.randint(0, max_n_masked_band)
                n_masked_frame = np.random.randint(0, max_n_masked_frame)
                masked_freq_band_start = np.random.randint(0, n_channel - n_masked_band)
                masked_frame_start = np.random.randint(0, n_frame - n_masked_frame)
                mel[i, masked_freq_band_start: masked_freq_band_start + n_masked_band] = fill_value
                mel[i, :, masked_frame_start: masked_frame_start + n_masked_frame] = fill_value
        return mel