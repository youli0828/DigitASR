"""Run fixed MFCC + SVM Hamming LOSO on AudioMNIST and synthetic SpokeN-100 Mandarin."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix

from digit_asr.features import FeatureConfig, features_from_file
from digit_asr.models import create_model
from digit_asr.speaker_loso import (
    loso_splits,
    prepare_speaker_manifest,
    read_speaker_manifest,
    validate_complete_dataset,
)

SOURCES = {
    "audiomnist": "https://github.com/soerenab/AudioMNIST",
    "spoken100_zh": "https://zenodo.org/records/10810044",
}


def evaluate_fold(speaker, train, test, features, labels):
    started = time.perf_counter()
    model = create_model("svm").fit(features[train], labels[train])
    predicted = model.predict(features[test])
    correct = int(np.sum(predicted == labels[test]))
    result = {
        "speaker": speaker, "train_count": len(train), "total": len(test),
        "correct": correct, "accuracy": correct / len(test),
        "confusion_matrix": confusion_matrix(labels[test], predicted, labels=range(10)).tolist(),
    }
    print(f"Held out {speaker}: {correct}/{len(test)} ({time.perf_counter()-started:.1f}s)",
          flush=True)
    return result, test.tolist(), predicted.tolist()


def run(dataset, root, output, jobs):
    config = FeatureConfig(window="hamming")
    manifest = prepare_speaker_manifest(root, dataset)
    rows = read_speaker_manifest(manifest)
    validate_complete_dataset(rows, dataset)
    fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    cache = Path("artifacts/speaker_loso") / f"{dataset}_{fingerprint}.npz"
    if cache.exists():
        with np.load(cache) as stored:
            if stored["config"].item() != json.dumps(asdict(config), sort_keys=True):
                raise ValueError("Cached feature configuration differs")
            features = stored["features"]
    else:
        vectors = []
        for i, row in enumerate(rows, 1):
            vectors.append(features_from_file(row["audio_path"], config))
            if i % 500 == 0:
                print(f"{dataset}: features {i}/{len(rows)}", flush=True)
        features = np.stack(vectors)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, features=features,
                            config=json.dumps(asdict(config), sort_keys=True))
    labels = np.array([r["label"] for r in rows])
    folds = Parallel(n_jobs=jobs)(
        delayed(evaluate_fold)(speaker, train, test, features, labels)
        for speaker, train, test in loso_splits(rows)
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{dataset}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "speaker", "label", "prediction", "sha256"])
        for _, indices, predicted in folds:
            for index, prediction in zip(indices, predicted):
                row = rows[index]
                writer.writerow([row["path"], row["speaker"], row["label"], prediction,
                                 row["sha256"]])
    per_speaker = [fold[0] for fold in folds]
    summary = {
        "dataset": dataset, "source": SOURCES[dataset],
        "language": "English" if dataset == "audiomnist" else "Mandarin Chinese",
        "synthetic": dataset == "spoken100_zh",
        "group_kind": "real speaker" if dataset == "audiomnist" else "TTS synthetic voice",
        "role": "PRIMARY" if dataset == "audiomnist" else "SECONDARY",
        "protocol": "Exhaustive LOSO: train N-1 speakers/voices, test the held-out group",
        "selection": "Fixed existing SVM C=10, RBF, gamma=scale; no test-speaker tuning",
        "scaling": "StandardScaler fitted exclusively on each fold's training data",
        "features": asdict(config), "model": "svm_hamming",
        "speaker_count": len(per_speaker), "count": len(rows),
        "dataset_sha256": fingerprint,
        "macro_accuracy": float(np.mean([fold["accuracy"] for fold in per_speaker])),
        "correct": sum(fold["correct"] for fold in per_speaker),
        "per_speaker": per_speaker,
        "environment": {
            "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in (
                "numpy", "scipy", "librosa", "soundfile", "scikit-learn", "joblib")},
        },
    }
    provenance = root / "source.json"
    if provenance.exists():
        summary["provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    (output / f"{dataset}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{dataset}: macro_accuracy={summary['macro_accuracy']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*SOURCES, "both"], default="both")
    parser.add_argument("--audiomnist", type=Path, default=Path("data/external/audiomnist"))
    parser.add_argument("--spoken100-zh", type=Path, default=Path("data/external/spoken100_zh"))
    parser.add_argument("--output", type=Path, default=Path("reports/speaker_loso"))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    for dataset in SOURCES if args.dataset == "both" else [args.dataset]:
        run(dataset, getattr(args, dataset), args.output, args.jobs)


if __name__ == "__main__":
    main()
