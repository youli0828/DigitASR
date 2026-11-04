import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from digit_asr.dataset import dataset_fingerprint, extract_split, read_manifest
from digit_asr.evaluate import evaluate_predictions, export_report
from digit_asr.features import FeatureConfig
from digit_asr.models import create_model, save_model
from digit_asr.pipeline import DigitRecognizer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/samples/manifest.csv"


@pytest.mark.parametrize("model_name", ["svm", "random_forest"])
def test_real_sample_train_save_predict(model_name, tmp_path):
    rows = read_manifest(MANIFEST)
    config = FeatureConfig()
    x_train, y_train, _ = extract_split(rows, "train", config)
    model = create_model(model_name).fit(x_train, y_train)
    path = tmp_path / "model.joblib"
    save_model(path, model, config, {"dataset_sha256": dataset_fingerprint(rows)})
    recognizer = DigitRecognizer(path)
    test_row = next(row for row in rows if row["split"] == "test")
    vector = extract_split([test_row], "test", config)[0]
    assert recognizer.predict(test_row["audio_path"]) == int(model.predict(vector)[0])
    assert 0 <= recognizer.predict(test_row["audio_path"]) <= 9
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "digit_asr.pipeline",
            str(test_row["audio_path"]),
            "--model",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["digit"] == recognizer.predict(test_row["audio_path"])


def test_dataset_split_and_fingerprint():
    rows = read_manifest(MANIFEST)
    assert len(rows) == 120
    assert len([row for row in rows if row["split"] == "train"]) == 80
    assert len(dataset_fingerprint(rows)) == 64
    train = {row["group"] for row in rows if row["split"] == "train"}
    test = {row["group"] for row in rows if row["split"] == "test"}
    assert not train & test
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        recorded = list(csv.DictReader(stream))
    assert all(row["sha256"] == original["sha256"] for row, original in zip(rows, recorded))


def test_reject_cross_split_source(tmp_path):
    rows = read_manifest(MANIFEST)
    manifest = tmp_path / "manifest.csv"
    (tmp_path / "a.wav").write_bytes(rows[0]["audio_path"].read_bytes())
    (tmp_path / "b.wav").write_bytes(rows[1]["audio_path"].read_bytes())
    manifest.write_text(
        "path,label,split,group\na.wav,0,train,same\nb.wav,1,test,same\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source group"):
        read_manifest(manifest)


def test_reject_cross_split_identical_audio(tmp_path):
    rows = read_manifest(MANIFEST)
    content = rows[0]["audio_path"].read_bytes()
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(content)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,label,split,group\na.wav,0,train,a\nb.wav,0,test,b\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Identical audio"):
        read_manifest(manifest)


def test_train_eval_scripts(tmp_path):
    model = tmp_path / "model.joblib"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            "--manifest",
            str(MANIFEST),
            "--model",
            "svm",
            "--window",
            "hann",
            "--output",
            str(model),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/eval.py"),
            "--manifest",
            str(MANIFEST),
            "--model",
            str(model),
            "--output",
            str(tmp_path / "evaluation"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["total"] == 40
    assert report["accuracy"] == report["correct"] / 40
    predictions = tmp_path / "evaluation/predictions.csv"
    with predictions.open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 40


def test_metrics_and_export(tmp_path):
    report = evaluate_predictions([0, 1, 2, 2], [0, 1, 1, 2])
    assert report["accuracy"] == 0.75
    assert report["correct"] == 3
    assert np.array(report["confusion_matrix"]).shape == (10, 10)
    assert report["confusion_matrix"][2][1] == 1
    export_report(report, tmp_path)
    assert json.loads((tmp_path / "metrics.json").read_text())["total"] == 4
    assert len((tmp_path / "confusion_matrix.csv").read_text().splitlines()) == 11
    with pytest.raises(ValueError):
        evaluate_predictions([], [])
