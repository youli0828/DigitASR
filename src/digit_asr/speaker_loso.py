"""Speaker/voice manifests and exhaustive leave-one-group-out evaluation."""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def prepare_speaker_manifest(root: str | Path, dataset: str) -> Path:
    root = Path(root).resolve()
    rows = []
    metadata = {}
    if dataset == "audiomnist":
        metadata = json.loads((root / "data/audioMNIST_meta.txt").read_text(encoding="utf-8"))
        paths = sorted((root / "data").glob("*/*.wav"))
    elif dataset == "spoken100_zh":
        paths = sorted(root.glob("*/*.wav"))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    for path in paths:
        speaker = path.parent.name
        if dataset == "audiomnist":
            match = re.fullmatch(r"([0-9])_([0-9]{2})_([0-9]+)", path.stem)
            if not match or match[2] != speaker or speaker not in metadata:
                raise ValueError(f"Invalid AudioMNIST filename/metadata: {path}")
            label = int(match[1])
        else:
            if path.stem not in {str(i) for i in range(10)}:
                raise ValueError(f"Expected only isolated digits 0-9: {path}")
            label = int(path.stem)
        rows.append([path.relative_to(root).as_posix(), label, speaker,
                     hashlib.sha256(path.read_bytes()).hexdigest()])
    if not rows:
        raise ValueError("Empty speaker dataset")
    manifest = root / "speaker_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "speaker", "sha256"])
        writer.writerows(rows)
    return manifest


def read_speaker_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty speaker manifest")
    seen, hashes = set(), {}
    for row in rows:
        audio = (path.parent / row["path"]).resolve()
        label = int(row["label"])
        if not audio.is_relative_to(path.parent):
            raise ValueError("Audio path must stay inside dataset")
        if audio in seen:
            raise ValueError("Duplicate audio path")
        if label not in range(10) or not row["speaker"].strip():
            raise ValueError("Expected labels 0-9 and a nonempty speaker/voice")
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError("Audio SHA-256 mismatch")
        if digest in hashes and hashes[digest] != (row["speaker"], label):
            raise ValueError("Identical audio crosses speakers/voices or labels")
        hashes[digest] = (row["speaker"], label)
        seen.add(audio)
        row.update(audio_path=audio, label=label)
    speakers = {r["speaker"] for r in rows}
    if len(speakers) < 2:
        raise ValueError("LOSO needs at least two speakers/voices")
    for speaker in speakers:
        if {r["label"] for r in rows if r["speaker"] == speaker} != set(range(10)):
            raise ValueError(f"Speaker/voice {speaker} must contain all ten digits")
    return rows


def loso_splits(rows):
    groups = np.array([row["speaker"] for row in rows])
    for speaker in sorted(set(groups)):
        yield str(speaker), np.flatnonzero(groups != speaker), np.flatnonzero(groups == speaker)


def validate_complete_dataset(rows, dataset):
    speakers, repetitions = {"audiomnist": (60, 50), "spoken100_zh": (32, 1)}[dataset]
    counts = Counter((r["speaker"], r["label"]) for r in rows)
    if len({r["speaker"] for r in rows}) != speakers or any(
        value != repetitions for value in counts.values()
    ) or len(counts) != speakers * 10:
        raise ValueError(f"Expected {speakers} speakers/voices x 10 digits x {repetitions}")
