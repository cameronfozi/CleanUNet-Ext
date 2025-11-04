# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import numpy as np

from scipy.io.wavfile import read as wavread
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from torchvision import datasets, models, transforms
import torchaudio

import yaml
from easydict import EasyDict as ed

from audiolib import audiowrite

from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# WATERMARKING FUNCTIONS
from echo_hiding import encode, decode, create_filter_bank
# CONFIG YAML
with open("echo_config.yaml", encoding="utf-8") as f:
    contents = yaml.load(f, Loader=yaml.FullLoader)
config = ed(contents)
filter_bank = create_filter_bank(config.kernel, config.delays, config.amplitude)

# WATERMARKING FLAG
WATERMARK = config.weight
print("WATERMARK WEIGHT IN DATASET:", WATERMARK)


def watermark(file_id, crop_length_sec):
    clean_audio, sample_rate = torchaudio.load(files[file_id][0])
    noisy_audio, sample_rate = torchaudio.load(files[file_id][1])
    clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
    assert len(clean_audio) == len(noisy_audio)

    crop_length = int(crop_length_sec * sample_rate)
    assert crop_length < len(clean_audio)

    # random crop
    start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
    clean_audio = clean_audio[start:(start + crop_length)]

    # ADD WATERMARKING BELOW
    if WATERMARK > 0:
        assert len(clean_audio) % config.win_size == 0, "Audio length must be an integer multiple of the window size" # make sure the audio length is an integer multiple of the window size
        assert len(clean_audio) // config.win_size == len(config.prior_symbols), "Number of windows in audio must match length of prior_symbols list"
        encoded_audio = encode(
            clean_audio.cpu().numpy(),
            config.prior_symbols,
            config.amplitude,
            config.delays,
            config.win_size,
            config.kernel,
            filters = filter_bank,
            hanning_factor = config.hanning_factor)

        clean_audio = torch.from_numpy(encoded_audio).to(clean_audio.device, clean_audio.dtype)

        noise, sample_rate = torchaudio.load(files[file_id][2])
        noise = noise.squeeze(0)
        noise_cropped = noise[start:(start + crop_length)] 
        noisy_audio = clean_audio + noise_cropped

    else:
        noisy_audio = noisy_audio[start:(start + crop_length)]

    # clean_audio, noisy_audio = clean_audio.unsqueeze(0), noisy_audio.unsqueeze(0)

    audiowrite(f'../watermarked_dns/training_set/clean/fileid_{file_id}.wav', clean_audio, 16000)
    audiowrite(f'../watermarked_dns/training_set/noisy/fileid_{file_id}.wav', noisy_audio, 16000)

    # return (clean_audio, noisy_audio, fileid)


N_clean = len(os.listdir(os.path.join('../dns/training_set/clean')))
files = [(
            os.path.join('../dns/training_set/clean', 'fileid_{}.wav'.format(i)),
            os.path.join('../dns/training_set/noisy', 'fileid_{}.wav'.format(i)),
            os.path.join('../dns/training_set/noise', 'fileid_{}.wav'.format(i))
        ) for i in range(N_clean)]


for file_id in tqdm(range(len(files))):
    watermark(file_id, crop_length_sec=10)

