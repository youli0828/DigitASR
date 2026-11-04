import numpy as np
import pytest
import soundfile as sf

from digit_asr.preprocess import (
    apply_window,
    frame_audio,
    load_audio,
    normalize_audio,
    pre_emphasis,
)


def test_pre_emphasis_preserves_first_sample():
    np.testing.assert_allclose(pre_emphasis(np.array([1, 2, 3]), 0.5), [1, 1.5, 2])


def test_frames_pad_last_and_short_frames():
    np.testing.assert_array_equal(
        frame_audio(np.arange(7), 4, 2), [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 0]]
    )
    np.testing.assert_array_equal(frame_audio(np.array([1, 2]), 4, 2), [[1, 2, 0, 0]])


def test_windows_are_distinct():
    frames = np.ones((2, 10))
    assert apply_window(frames, "hann")[0, 0] == 0
    assert apply_window(frames, "hamming")[0, 0] == pytest.approx(0.08)
    with pytest.raises(ValueError):
        apply_window(frames, "unknown")


@pytest.mark.parametrize("audio", [np.zeros(40), np.array([]), np.array([np.nan]), np.ones((3, 2))])
def test_invalid_audio(audio):
    with pytest.raises(ValueError):
        normalize_audio(audio)


def test_stereo_resample_and_trim(tmp_path):
    wave = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack([wave, wave * 0.5]), 8000)
    audio = load_audio(path)
    assert audio.ndim == 1
    assert 15900 < len(audio) <= 16000
    assert np.max(np.abs(audio)) == pytest.approx(1)
    spectrum = np.abs(np.fft.rfft(audio))
    frequency = np.fft.rfftfreq(len(audio), 1 / 16000)[spectrum.argmax()]
    assert frequency == pytest.approx(440, abs=2)
