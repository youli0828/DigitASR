from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    audio, original_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if original_rate != sample_rate:
        divisor = gcd(original_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, original_rate // divisor)
    return normalize_audio(audio)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must be a nonempty finite mono array")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        raise ValueError("Audio is silent")
    # Remove only leading/trailing near-silence; retain internal consonants and pauses.
    active = np.flatnonzero(np.abs(audio) > peak * 0.01)
    audio = audio[active[0] : active[-1] + 1]
    return audio / peak


def pre_emphasis(audio: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    if not 0 <= coefficient < 1:
        raise ValueError("coefficient must be in [0, 1)")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Expected nonempty mono audio")
    return np.concatenate((audio[:1], audio[1:] - coefficient * audio[:-1]))


def frame_audio(audio: np.ndarray, frame_length: int = 400, hop_length: int = 160):
    audio = np.asarray(audio)
    if audio.ndim != 1 or not audio.size or frame_length <= 0 or hop_length <= 0:
        raise ValueError("Expected mono audio and positive frame/hop lengths")
    count = 1 + max(0, int(np.ceil((len(audio) - frame_length) / hop_length)))
    padded = np.pad(audio, (0, (count - 1) * hop_length + frame_length - len(audio)))
    starts = np.arange(count)[:, None] * hop_length
    return padded[starts + np.arange(frame_length)]


def apply_window(frames: np.ndarray, window: str = "hamming") -> np.ndarray:
    if window not in ("hann", "hamming"):
        raise ValueError("window must be hann or hamming")
    weights = np.hanning(frames.shape[-1]) if window == "hann" else np.hamming(frames.shape[-1])
    return frames * weights
