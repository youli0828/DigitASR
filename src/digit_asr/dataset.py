import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np

from digit_asr.features import FeatureConfig, features_from_file

DIGIT_WORDS = dict(
    zip(
        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
        range(10),
    )
)


def gitcode_digit_label(path: str | Path) -> int:
    path = Path(path)
    match = re.fullmatch(r"([a-z]+|[0-9])-([0-9]+)", path.stem.lower())
    if match and match[1] in DIGIT_WORDS:
        return DIGIT_WORDS[match[1]]
    if match and match[1].isdigit():
        return int(match[1])
    if path.parent.name in DIGIT_WORDS:
        return DIGIT_WORDS[path.parent.name]
    if path.parent.name in tuple(str(n) for n in range(10)):
        return int(path.parent.name)
    raise ValueError(f"Cannot infer digit label: {path}")


def prepare_gitcode_manifest(root: str | Path, seed: int = 42) -> Path:
    from sklearn.model_selection import StratifiedGroupKFold

    root = Path(root).resolve()
    paths = sorted(root.rglob("*.wav"))
    labels = [gitcode_digit_label(path) for path in paths]
    if Counter(labels) != Counter({digit: 25 for digit in range(10)}):
        raise ValueError("Expected exactly 250 WAV files, 25 per digit")
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    # Shuffle group IDs explicitly: sklearn <1.8 shuffles counts without remapping IDs.
    shuffled_hashes = np.random.RandomState(seed).permutation(np.unique(hashes))
    group_ids = {digest: index for index, digest in enumerate(shuffled_hashes)}
    groups = [group_ids[digest] for digest in hashes]
    _, test = next(StratifiedGroupKFold(5, shuffle=False).split(paths, labels, groups))
    test = set(test)
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "split", "group", "sha256"])
        writer.writeheader()
        for index, (path, label) in enumerate(zip(paths, labels)):
            relative = path.relative_to(root).as_posix()
            writer.writerow(
                {
                    "path": relative,
                    "label": label,
                    "split": "test" if index in test else "train",
                    # Audio identity only: the source provides no speaker metadata.
                    "group": hashes[index],
                    "sha256": hashes[index],
                }
            )
    read_manifest(manifest)
    return manifest


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest")
    seen_paths, seen_hashes, groups = set(), {}, {}
    for row in rows:
        label = int(row["label"])
        split = row["split"]
        group = row["group"]
        audio_path = (path.parent / row["path"]).resolve()
        if label not in range(10) or split not in ("train", "test") or not group:
            raise ValueError("Expected labels 0-9, train/test split, and nonempty group")
        if not audio_path.is_relative_to(path.parent):
            raise ValueError("Audio paths must remain inside the dataset directory")
        if audio_path in seen_paths:
            raise ValueError("Duplicate audio path")
        seen_paths.add(audio_path)
        digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        if row.get("sha256") and row["sha256"] != digest:
            raise ValueError(f"Audio differs from its recorded SHA-256: {row['path']}")
        if digest in seen_hashes and seen_hashes[digest] != split:
            raise ValueError("Identical audio occurs in train and test")
        if group in groups and groups[group] != split:
            raise ValueError("A source group occurs in both train and test")
        seen_hashes[digest], groups[group] = split, split
        row.update(label=label, audio_path=audio_path, sha256=digest)
    for split in ("train", "test"):
        if {row["label"] for row in rows if row["split"] == split} != set(range(10)):
            raise ValueError(f"{split} must contain all ten digits")
    return rows


def dataset_fingerprint(rows: list[dict]) -> str:
    payload = "\n".join(
        f"{r['path']}:{r['label']}:{r['split']}:{r['group']}:{r['sha256']}" for r in rows
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_split(rows: list[dict], split: str, config: FeatureConfig):
    selected = [row for row in rows if row["split"] == split]
    return (
        np.stack([features_from_file(row["audio_path"], config) for row in selected]),
        np.array([row["label"] for row in selected]),
        selected,
    )
