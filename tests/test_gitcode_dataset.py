from collections import Counter

import pytest

from digit_asr.dataset import (
    DIGIT_WORDS,
    dataset_fingerprint,
    gitcode_digit_label,
    prepare_gitcode_manifest,
    read_manifest,
)


@pytest.mark.parametrize("word,digit", DIGIT_WORDS.items())
def test_gitcode_labels(word, digit):
    assert gitcode_digit_label(f"{word}-25.wav") == digit
    assert gitcode_digit_label(f"{digit}/recording.wav") == digit


def test_gitcode_reject_unknown_label():
    with pytest.raises(ValueError, match="Cannot infer"):
        gitcode_digit_label("unknown-1.wav")


@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_gitcode_stratification_and_duplicate_isolation(tmp_path, seed):
    for word in DIGIT_WORDS:
        for index in range(1, 26):
            # Two filenames deliberately share content within each digit.
            content = f"{word}-{1 if index == 2 else index}".encode()
            (tmp_path / f"{word}-{index}.wav").write_bytes(content)
    manifest = prepare_gitcode_manifest(tmp_path, seed=seed)
    rows = read_manifest(manifest)
    assert len(rows) == 250
    assert len({r["sha256"] for r in rows}) == 240
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    assert Counter(r["label"] for r in train) == Counter({i: 20 for i in range(10)})
    assert Counter(r["label"] for r in test) == Counter({i: 5 for i in range(10)})
    assert not {r["sha256"] for r in train} & {r["sha256"] for r in test}
    fingerprint = dataset_fingerprint(rows)
    prepare_gitcode_manifest(tmp_path, seed=seed)
    assert dataset_fingerprint(read_manifest(manifest)) == fingerprint
    prepare_gitcode_manifest(tmp_path, seed=seed + 1)
    assert dataset_fingerprint(read_manifest(manifest)) != fingerprint


def test_gitcode_reject_incomplete_dataset(tmp_path):
    (tmp_path / "zero-1.wav").write_bytes(b"sample")
    with pytest.raises(ValueError, match="250 WAV"):
        prepare_gitcode_manifest(tmp_path)
