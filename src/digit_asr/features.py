from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from digit_asr.preprocess import (
    apply_window,
    frame_audio,
    load_audio,
    normalize_audio,
    pre_emphasis,
)


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    frame_length: int = 400
    hop_length: int = 160
    n_fft: int = 512
    n_mels: int = 40
    n_mfcc: int = 13
    window: str = "hamming"
    pre_emphasis: float = 0.97
    temporal_bins: int = 4
    sequence_frames: int | None = None

    def __post_init__(self):
        if self.sequence_frames is not None and self.sequence_frames < 4:
            raise ValueError("sequence_frames must be at least four")
        if self.window not in ("hann", "hamming"):
            raise ValueError("window must be hann or hamming")
        if not (0 < self.n_mfcc <= self.n_mels <= self.n_fft // 2 + 1):
            raise ValueError("Invalid MFCC/mel dimensions")
        if min(self.sample_rate, self.frame_length, self.hop_length, self.temporal_bins) <= 0:
            raise ValueError("Lengths and sample rate must be positive")
        if self.n_fft < self.frame_length or not 0 <= self.pre_emphasis < 1:
            raise ValueError("Invalid FFT length or pre-emphasis")


def extract_features(audio: np.ndarray, config: FeatureConfig | None = None) -> np.ndarray:
    config = config or FeatureConfig()
    audio = pre_emphasis(normalize_audio(audio), config.pre_emphasis)
    frames = apply_window(frame_audio(audio, config.frame_length, config.hop_length), config.window)
    power = np.abs(np.fft.rfft(frames, n=config.n_fft, axis=1)) ** 2 / config.n_fft
    mel_basis = librosa.filters.mel(sr=config.sample_rate, n_fft=config.n_fft, n_mels=config.n_mels)
    log_mel = librosa.power_to_db(mel_basis @ power.T, ref=1.0)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=config.n_mfcc)
    if config.sequence_frames is not None:
        indices = np.linspace(0, mfcc.shape[1] - 1, config.sequence_frames)
        return np.array(
            [np.interp(indices, np.arange(mfcc.shape[1]), row) for row in mfcc],
            dtype=np.float32,
        )
    # Pool over normalized time so word order survives without fixed-length audio.
    indices = np.linspace(0, mfcc.shape[1] - 1, max(config.temporal_bins, mfcc.shape[1]))
    aligned = np.array([np.interp(indices, np.arange(mfcc.shape[1]), row) for row in mfcc])
    pooled = [part.mean(axis=1) for part in np.array_split(aligned, config.temporal_bins, axis=1)]
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), *pooled]).astype(np.float32)


def features_from_file(path: str | Path, config: FeatureConfig | None = None) -> np.ndarray:
    config = config or FeatureConfig()
    return extract_features(load_audio(path, config.sample_rate), config)
