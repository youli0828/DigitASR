"""Measure fixed CNN and SVM Hamming baselines on the same real GitCode holdout."""

import argparse
import csv
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from digit_asr.dataset import (
    dataset_fingerprint,
    extract_split,
    prepare_gitcode_manifest,
    read_manifest,
)
from digit_asr.evaluate import evaluate_predictions, export_report
from digit_asr.features import FeatureConfig
from digit_asr.models import create_model, save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/external/gitcode_digits"))
    parser.add_argument("--output", type=Path, default=Path("reports/cnn"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_manifest(prepare_gitcode_manifest(args.data, args.seed))
    fingerprint = dataset_fingerprint(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "split.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "split", "group", "sha256"])
        writer.writerows(
            [row[key] for key in ("path", "label", "split", "group", "sha256")] for row in rows
        )
    summaries = []
    for name in ("svm_hamming", "cnn"):
        config = FeatureConfig(sequence_frames=64 if name == "cnn" else None)
        x_train, y_train, train_rows = extract_split(rows, "train", config)
        x_test, y_test, test_rows = extract_split(rows, "test", config)
        model = create_model(name, args.seed)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        training_seconds = time.perf_counter() - started
        predictions = model.predict(x_test)
        metadata = {
            "model": name,
            "seed": args.seed,
            "dataset_sha256": fingerprint,
            "features": asdict(config),
            "train_count": len(y_train),
            "train_paths": [row["path"] for row in train_rows],
        }
        report = {
            **evaluate_predictions(y_test, predictions),
            **metadata,
            "test_paths": [row["path"] for row in test_rows],
            "training_seconds": training_seconds,
        }
        if name == "cnn":
            report["training"] = {
                "epochs": model.epochs,
                "batch_size": model.batch_size,
                "learning_rate": model.learning_rate,
                "weight_decay": 1e-4,
                "device": "cpu",
                "threads": 1,
                "parameter_count": model.parameter_count,
                "normalization": "MFCC channel mean/std fitted on train only",
                "selection": "Fixed architecture and 100 epochs; no test tuning or selection",
            }
        export_report(report, args.output / name)
        with (args.output / name / "predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "label", "prediction"])
            writer.writerows(
                [row["path"], int(label), int(prediction)]
                for row, label, prediction in zip(test_rows, y_test, predictions)
            )
        save_model(Path("artifacts/cnn") / name / "model.joblib", model, config, metadata)
        summary = {key: report[key] for key in ("model", "accuracy", "correct", "total")}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    summary = {
        "source": "https://gitcode.com/Universal-Tool/9baeb",
        "seed": args.seed,
        "dataset_sha256": fingerprint,
        "split_method": "prepare_gitcode_manifest: seed-shuffled SHA256 IDs, SGKF(5) first fold",
        "speaker_disjoint": False,
        "limitation": "No verified speaker IDs; this is not cross-speaker evaluation",
        "experiments": summaries,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "scikit-learn", "librosa", "joblib")
            },
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
