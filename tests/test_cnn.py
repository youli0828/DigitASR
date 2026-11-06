import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch", exc_type=ImportError)

from digit_asr.dataset import extract_split, read_manifest
from digit_asr.features import FeatureConfig
from digit_asr.models import create_model, load_model, save_model
from digit_asr.pipeline import DigitRecognizer

ROOT = Path(__file__).resolve().parents[1]


def test_cnn_train_save_load_and_determinism(tmp_path):
    x = np.random.default_rng(7).normal(size=(20, 13, 16)).astype(np.float32)
    y = np.repeat(np.arange(10), 2)
    model = create_model("cnn")
    model.epochs = 3
    model.fit(x, y)
    assert model.parameter_count < 10_000
    np.testing.assert_allclose(model.mean_, x.mean(axis=(0, 2), keepdims=True))
    assert not model.network_.training
    duplicate = create_model("cnn")
    duplicate.epochs = 3
    duplicate.fit(x, y)
    for first, second in zip(model.network_.parameters(), duplicate.network_.parameters()):
        np.testing.assert_array_equal(first.detach().numpy(), second.detach().numpy())
    path = tmp_path / "cnn.joblib"
    save_model(path, model, FeatureConfig(sequence_frames=16), {"model": "cnn"})
    restored = load_model(path)["estimator"]
    np.testing.assert_array_equal(model.predict(x), restored.predict(x))
    assert restored.predict(x).shape == (20,)
    assert set(restored.predict(x)) <= set(range(10))


def test_cnn_offline_train_eval_and_recognizer(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "split", "group"])
        for digit in range(10):
            for index, split in enumerate(("train", "train", "test")):
                name = f"{digit}_{index}.wav"
                time = np.arange(1600 + index * 100) / 16000
                audio = np.sin(2 * np.pi * (250 + digit * 70 + index) * time)
                sf.write(tmp_path / name, audio, 16000)
                writer.writerow([name, digit, split, f"{digit}_{index}"])
    model_path = tmp_path / "model.joblib"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            "--model",
            "cnn",
            "--manifest",
            str(manifest),
            "--output",
            str(model_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = tmp_path / "evaluation"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/eval.py"),
            "--model",
            str(model_path),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output / "metrics.json").read_text())
    assert report["total"] == 10
    assert report["accuracy"] == report["correct"] / 10
    assert np.asarray(report["confusion_matrix"]).sum() == 10
    rows = read_manifest(manifest)
    bundle = load_model(model_path)
    x_test, _, selected = extract_split(rows, "test", bundle["config"])
    assert x_test.shape == (10, 13, 64)
    recognizer = DigitRecognizer(model_path)
    assert recognizer.predict(selected[0]["audio_path"]) == bundle["estimator"].predict(x_test)[0]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "digit_asr.pipeline",
            str(selected[0]["audio_path"]),
            "--model",
            str(model_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["digit"] == recognizer.predict(selected[0]["audio_path"])
