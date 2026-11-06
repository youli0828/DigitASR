import csv
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from digit_asr.speaker_loso import (
    loso_splits,
    prepare_speaker_manifest,
    read_speaker_manifest,
    validate_complete_dataset,
)


@pytest.fixture(params=["audiomnist", "spoken100_zh"])
def tiny_dataset(tmp_path, request):
    dataset = request.param
    if dataset == "audiomnist":
        (tmp_path / "data").mkdir()
        (tmp_path / "data/audioMNIST_meta.txt").write_text(
            json.dumps({"01": {"age": 20}, "02": {"age": 30}, "03": {"age": 40}}),
            encoding="utf-8",
        )
    for speaker in ("01", "02", "03"):
        directory = tmp_path / "data" / speaker if dataset == "audiomnist" else tmp_path / speaker
        directory.mkdir()
        for label in range(10):
            name = f"{label}_{speaker}_0.wav" if dataset == "audiomnist" else f"{label}.wav"
            (directory / name).write_bytes(f"fixture-{speaker}-{label}".encode())
    return tmp_path, dataset


def test_manifest_and_exhaustive_isolated_splits(tiny_dataset):
    root, dataset = tiny_dataset
    rows = read_speaker_manifest(prepare_speaker_manifest(root, dataset))
    assert len(rows) == 30
    seen_test = []
    for speaker, train, test in loso_splits(rows):
        assert len(train) == 20 and len(test) == 10
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(range(30))
        assert all(rows[i]["speaker"] != speaker for i in train)
        assert all(rows[i]["speaker"] == speaker for i in test)
        assert {rows[i]["label"] for i in test} == set(range(10))
        seen_test.extend(test)
    assert sorted(seen_test) == list(range(30))
    with pytest.raises(ValueError, match="Expected"):
        validate_complete_dataset(rows, dataset)


@pytest.mark.parametrize("problem", ["hash", "duplicate", "escape", "cross_speaker", "missing"])
def test_reject_corrupt_manifests(tiny_dataset, problem):
    root, dataset = tiny_dataset
    manifest = prepare_speaker_manifest(root, dataset)
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {
        "hash": "SHA-256", "duplicate": "Duplicate", "escape": "inside dataset",
        "cross_speaker": "Identical audio", "missing": "all ten digits",
    }
    if problem == "hash":
        (root / rows[0]["path"]).write_bytes(b"changed")
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "escape":
        rows[0]["path"] = "../outside.wav"
    elif problem == "cross_speaker":
        other = next(r for r in rows if r["speaker"] != rows[0]["speaker"])
        (root / other["path"]).write_bytes((root / rows[0]["path"]).read_bytes())
        other["sha256"] = rows[0]["sha256"]
    else:
        rows.pop(0)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "speaker", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=expected[problem]):
        read_speaker_manifest(manifest)


def test_audiomnist_requires_matching_metadata(tmp_path):
    (tmp_path / "data/01").mkdir(parents=True)
    (tmp_path / "data/audioMNIST_meta.txt").write_text('{"01": {}}')
    (tmp_path / "data/01/0_02_0.wav").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="filename/metadata"):
        prepare_speaker_manifest(tmp_path, "audiomnist")


@pytest.mark.parametrize("name,expected", [
    ("SpokeN-100/mandarin_numbers/Serena/0.wav", ("Serena", 0)),
    ("SpokeN-100/mandarin_numbers/Arnold/9.wav", ("Arnold", 9)),
    ("SpokeN-100/mandarin_numbers/Arnold/10.wav", None),
    ("SpokeN-100/mandarin_numbers/Arnold/99.wav", None),
    ("SpokeN-100/english_numbers/Arnold/1.wav", None),
    ("SpokeN-100/mandarin_numbers/Arnold/", None),
])
def test_zenodo_subset_member_mapping(name, expected):
    script = Path(__file__).resolve().parents[1] / "scripts/download_spoken100_zh.py"
    spec = importlib.util.spec_from_file_location("download_spoken100_zh", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.mandarin_digit_member(name) == expected


def test_scaling_uses_training_groups_only():
    from digit_asr.models import create_model

    rows = [{"speaker": str(i // 10)} for i in range(30)]
    features = np.arange(60, dtype=float).reshape(30, 2)
    labels = np.tile(np.arange(10), 3)
    for _, train, test in loso_splits(rows):
        model = create_model("svm").fit(features[train], labels[train])
        np.testing.assert_allclose(model[0].mean_, features[train].mean(axis=0))
        assert model[0].n_samples_seen_ == len(train)
        assert len(model.predict(features[test])) == 10


def test_range_downloader_extracts_only_digits_and_resumes(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/download_spoken100_zh.py"
    spec = importlib.util.spec_from_file_location("download_spoken100_zh", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SpokeN-100/english_numbers/Other/0.wav", b"x" * 70000)
        for voice in range(32):
            for digit in range(11):
                archive.writestr(
                    f"SpokeN-100/mandarin_numbers/Voice{voice}/{digit}.wav",
                    f"voice-{voice}-digit-{digit}".encode(), compress_type=zipfile.ZIP_DEFLATED,
                )
    data = buffer.getvalue()
    record = {"files": [{"key": "SpokeN-100.zip", "size": len(data),
                         "checksum": "fixture"}], "metadata": {"license": {"id": "cc-by-4.0"}}}
    monkeypatch.setattr(module.urllib.request, "urlopen",
                        lambda *args, **kwargs: io.BytesIO(json.dumps(record).encode()))
    requests = []

    def fixture_range(start, end):
        requests.append((start, end))
        return data[start:end + 1]

    monkeypatch.setattr(module, "fetch_range", fixture_range)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(tmp_path)])
    module.main()
    assert len(list(tmp_path.glob("*/*.wav"))) == 320
    assert (tmp_path / "Voice0/0.wav").read_bytes() == b"voice-0-digit-0"
    assert not (tmp_path / "Voice0/10.wav").exists()
    assert json.loads((tmp_path / "source.json").read_text())["synthetic"] is True
    requests.clear()
    module.main()
    assert len(requests) == 2  # Only EOCD and central directory; verified audio is reused.
