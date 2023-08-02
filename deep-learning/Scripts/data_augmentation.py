import librosa
import torch
import torchaudio
import torchaudio.functional as F

print(torch.__version__)
print(torchaudio.__version__)
from torchaudio.utils import download_asset
import numpy as np
import os
import random
from torchaudio.utils import download_asset

NOISE_DIR = r"/zhome/bd/4/181258/code/deep_learning/FSDKaggle2018.audio_test/"

import evaluate
from datasets import load_dataset, Audio
import torch
import torchaudio.transforms as T

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
print(dev)
device = torch.device(dev)  

librispeech = load_dataset('librispeech_asr', 'clean', split="test")
# librispeech = librispeech.cast_column("audio", Audio(sampling_rate=16000))
print("Dataset loaded")


def augmentate_audio(batch):
    speech_sample = batch["audio"]
    waveform_speech, sample_rate_speech = torch.from_numpy(speech_sample["array"]), speech_sample["sampling_rate"]
    waveform_speech = waveform_speech.to(dev)
    
    while True:
        waveform_noise, sample_rate_noise = torchaudio.load(NOISE_DIR + random.choice(os.listdir(NOISE_DIR)))
        waveform_noise = waveform_noise[0].to(dev)
        if len(waveform_noise) / sample_rate_noise > 2.0:
            break

    resampler = T.Resample(sample_rate_noise, sample_rate_speech, dtype=waveform_noise.dtype).cuda()
    waveform_noise = resampler(waveform_noise)

    if len(waveform_noise) < len(waveform_speech):
        waveform_noise = torch.cat((int(len(waveform_speech) / len(waveform_noise)) + 1)*[waveform_noise])
        
    waveform_noise = waveform_noise[: waveform_speech.shape[0]]

    speech_rms = waveform_speech.norm(p=2)
    noise_rms = waveform_noise.norm(p=2)
    
    snr_db =  random.choice([20, 10, 3])
    snr = 10 ** (snr_db / 20)
    scale = snr * noise_rms / speech_rms
    
    batch["audio"]["array"] = (scale * waveform_speech + waveform_noise) / 2
    return batch


librispeech_augmentated = librispeech.map(augmentate_audio)
librispeech_augmentated.save_to_disk("librispeech_augmentated_test.hf")