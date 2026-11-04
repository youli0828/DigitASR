import numpy as np
import pytest

from digit_asr.features import FeatureConfig, extract_features


@pytest.mark.parametrize("window", ["hann", "hamming"])
def test_mfcc_finite_and_amplitude_invariant(window):
    config = FeatureConfig(window=window)
    wave = np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)
    features = extract_features(wave, config)
    assert features.shape == (78,)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features, extract_features(wave * 0.1, config), atol=1e-3)
    other = extract_features(np.sin(2 * np.pi * 1500 * np.arange(8000) / 16000), config)
    assert np.linalg.norm(features - other) > 1


def test_short_audio_and_invalid_config():
    assert np.isfinite(extract_features(np.array([0.2, 1.0]))).all()
    with pytest.raises(ValueError):
        FeatureConfig(n_fft=100)
    with pytest.raises(ValueError):
        FeatureConfig(window="rectangular")
