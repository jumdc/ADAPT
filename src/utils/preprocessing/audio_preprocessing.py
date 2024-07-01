"""Audio utils for preprocessing."""

import warnings

import torch
import librosa
import torchaudio
import torchvision
import numpy as np

EPS = torch.finfo(torch.float).eps
warnings.filterwarnings("ignore")


def audio_preprocess(cfg, file_path=None, data=None):
    """
    Audio preprocessing function.

    Parameters
    ----------
    cfg : DictConfig
        Configuration file.
    file_path : str
        Path to the audio file.
    data : torch.Tensor
        Audio data.

    Returns
    -------
    torch.Tensor
        Preprocessed audio data.
    """
    if file_path is not None:
        waveform, freq = librosa.load(file_path, mono=True, sr=None)
    else:
        waveform, freq = data, cfg.audio_hyperparams.audio_freq
    sample_rate = cfg.audio_hyperparams.sample_rate
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    resample = torchaudio.transforms.Resample(orig_freq=freq, new_freq=sample_rate)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.audio_hyperparams.sample_rate,
        n_fft=cfg.audio_hyperparams.n_fft,
        win_length=cfg.audio_hyperparams.win_length,
        hop_length=cfg.audio_hyperparams.hop_length,
        f_min=cfg.audio_hyperparams.f_min,
        f_max=cfg.audio_hyperparams.f_max,
        n_mels=cfg.audio_hyperparams.n_mels,
    )
    resize = (64, 469) if cfg.audio_hyperparams.window_audio == 480000 else (64, 938)
    resize = torchvision.transforms.Resize(resize)
    waveform = resample(waveform) if freq != sample_rate else waveform
    audio = (
        (mel_spec(waveform) + EPS).log() - cfg.audio_hyperparams.stats[0]
    ) / cfg.audio_hyperparams.stats[1]
    audio = resize(audio.unsqueeze(0))
    return audio
