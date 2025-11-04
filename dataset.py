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

torch.autograd.set_detect_anomaly(True)

# WATERMARKING FUNCTIONS
from watermark.echo_hiding import encode, decode, create_filter_bank
# CONFIG YAML
with open("watermark/echo_config.yaml", encoding="utf-8") as f:
    contents = yaml.load(f, Loader=yaml.FullLoader)
config = ed(contents)
filter_bank = create_filter_bank(config.kernel, config.delays, config.amplitude)

# WATERMARKING FLAG
WATERMARK = config.weight
print("WATERMARK WEIGHT IN DATASET:", WATERMARK)


class CleanNoisyPairDataset(Dataset):
    """
    Create a Dataset of clean and noisy audio pairs. 
    Each element is a tuple of the form (clean waveform, noisy waveform, file_id)
    """
    
    def __init__(self, root='./', subset='training', crop_length_sec=0):
        super(CleanNoisyPairDataset).__init__()

        assert subset is None or subset in ["training", "testing"]
        self.crop_length_sec = crop_length_sec
        self.subset = subset

        N_clean = len(os.listdir(os.path.join(root, 'training_set/clean')))
        N_noisy = len(os.listdir(os.path.join(root, 'training_set/noisy')))
        assert N_clean == N_noisy

        if subset == "training":
            self.files = [(os.path.join(root, 'training_set/clean', 'fileid_{}.wav'.format(i)),
                           os.path.join(root, 'training_set/noisy', 'fileid_{}.wav'.format(i)),
                           os.path.join(root, 'training_set/noise', 'fileid_{}.wav'.format(i))) for i in range(N_clean)]
        
        elif subset == "testing":
            sortkey = lambda name: '_'.join(name.split('_')[-2:])  # specific for dns due to test sample names
            _p = os.path.join(root, 'datasets/test_set/synthetic/no_reverb')  # path for DNS
            
            clean_files = os.listdir(os.path.join(_p, 'clean'))
            noisy_files = os.listdir(os.path.join(_p, 'noisy'))
            
            clean_files.sort(key=sortkey)
            noisy_files.sort(key=sortkey)

            self.files = []
            for _c, _n in zip(clean_files, noisy_files):
                assert sortkey(_c) == sortkey(_n)
                self.files.append((os.path.join(_p, 'clean', _c), 
                                   os.path.join(_p, 'noisy', _n)))
            self.crop_length_sec = 0

        else:
            raise NotImplementedError

    def __getitem__(self, n):
        fileid = self.files[n]
        clean_audio, sample_rate = torchaudio.load(fileid[0])
        noisy_audio, sample_rate = torchaudio.load(fileid[1])
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
        assert len(clean_audio) == len(noisy_audio)

        crop_length = int(self.crop_length_sec * sample_rate)
        assert crop_length < len(clean_audio)

        # random crop
        if self.subset != 'testing' and crop_length > 0:
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

                noise, sample_rate = torchaudio.load(fileid[2])
                noise = noise.squeeze(0)
                noise_cropped = noise[start:(start + crop_length)] 
                noisy_audio = clean_audio + noise_cropped
            # ADD WATERMARKING ABOVE

            else:
                noisy_audio = noisy_audio[start:(start + crop_length)]
        
        clean_audio, noisy_audio = clean_audio.unsqueeze(0), noisy_audio.unsqueeze(0)
        return (clean_audio, noisy_audio, fileid)

    def __len__(self):
        return len(self.files)


def load_CleanNoisyPairDataset(root, subset, crop_length_sec, batch_size, sample_rate, num_gpus=1):
    """
    Get dataloader with distributed sampling
    """
    dataset = CleanNoisyPairDataset(root=root, subset=subset, crop_length_sec=crop_length_sec)                                                       
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False}

    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)
        
    return dataloader


if __name__ == '__main__':
    import json
    with open('./configs/DNS-large-full.json') as f:
        data = f.read()
    config = json.loads(data)
    trainset_config = config["trainset_config"]

    trainloader = load_CleanNoisyPairDataset(**trainset_config, subset='training', batch_size=2, num_gpus=1)
    testloader = load_CleanNoisyPairDataset(**trainset_config, subset='testing', batch_size=2, num_gpus=1)
    print(len(trainloader), len(testloader))

    for clean_audio, noisy_audio, fileid in trainloader: 
        clean_audio = clean_audio.cuda()
        noisy_audio = noisy_audio.cuda()
        print(clean_audio.shape, noisy_audio.shape, fileid)
        break
    