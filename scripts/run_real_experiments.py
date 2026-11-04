"""Evaluate fixed baselines on GitCode recordings, with training-only 5-fold CV."""

import argparse
import csv
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold

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
    parser.add_argument("--output", type=Path, default=Path("reports/real"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare_gitcode_manifest(args.data, args.seed)
    rows = read_manifest(manifest)
    fingerprint = dataset_fingerprint(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "split.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "label", "split", "sha256"])
        writer.writerows([r["audio_path"].name, r["label"], r["split"], r["sha256"]] for r in rows)
    summaries = []
    for window in ("hamming", "hann"):
        config = FeatureConfig(window=window)
        x_train, y_train, train_rows = extract_split(rows, "train", config)
        x_test, y_test, test_rows = extract_split(rows, "test", config)
        folds = list(
            StratifiedGroupKFold(5, shuffle=True, random_state=args.seed).split(
                x_train, y_train, [r["sha256"] for r in train_rows]
            )
        )
        for name in ("svm", "random_forest"):
            model = create_model(name, args.seed)
            cv_reports, cv_predictions = [], []
            for fold, (train, valid) in enumerate(folds, 1):
                estimator = clone(model).fit(x_train[train], y_train[train])
                predicted = estimator.predict(x_train[valid])
                measured = evaluate_predictions(y_train[valid], predicted)
                cv_reports.append({k: measured[k] for k in ("accuracy", "correct", "total")})
                cv_predictions.extend(
                    [fold, train_rows[i]["audio_path"].name, int(y_train[i]), int(p)]
                    for i, p in zip(valid, predicted)
                )
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            report = evaluate_predictions(y_test, predicted)
            metadata = {
                "model": name,
                "seed": args.seed,
                "features": asdict(config),
                "train_count": len(y_train),
                "dataset_sha256": fingerprint,
                "train_paths": [r["path"] for r in train_rows],
            }
            report.update(metadata)
            report["cv"] = {
                "scope": "200 training recordings only; held-out test excluded",
                "folds": cv_reports,
                "mean_accuracy": float(np.mean([r["accuracy"] for r in cv_reports])),
                "correct": sum(r["correct"] for r in cv_reports),
                "total": len(y_train),
            }
            report["test_paths"] = [r["path"] for r in test_rows]
            experiment = f"{name}_{window}"
            output = args.output / experiment
            export_report(report, output)
            for filename, header, values in (
                (
                    "predictions.csv",
                    ["file", "label", "prediction"],
                    [
                        [r["audio_path"].name, int(y), int(p)]
                        for r, y, p in zip(test_rows, y_test, predicted)
                    ],
                ),
                ("cv_predictions.csv", ["fold", "file", "label", "prediction"], cv_predictions),
            ):
                with (output / filename).open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header)
                    writer.writerows(values)
            save_model(
                Path("artifacts/real") / experiment / "model.joblib", model, config, metadata
            )
            summary = {
                "model": name,
                "window": window,
                "train_count": len(y_train),
                **{k: report[k] for k in ("accuracy", "correct", "total", "cv")},
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    best = max(summaries, key=lambda s: s["cv"]["mean_accuracy"])
    best_name = f"{best['model']}_{best['window']}"
    report = json.loads((args.output / best_name / "metrics.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ConfusionMatrixDisplay(np.array(report["confusion_matrix"]), display_labels=range(10)).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title(
        f"GitCode recordings: {best_name}\nHeld-out correct: {best['correct']}/{best['total']}"
    )
    image = Path("docs/images/real_best_confusion_matrix.png")
    image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image, dpi=160)
    plt.close(fig)
    result = {
        "source": "https://gitcode.com/Universal-Tool/9baeb",
        "seed": args.seed,
        "dataset_sha256": fingerprint,
        "count": len(rows),
        "unique_audio_sha256": len({r["sha256"] for r in rows}),
        "split_method": "First fold of StratifiedGroupKFold(5, shuffle=True, random_state=seed)",
        "grouping": "Audio SHA-256; duplicates never cross holdout or CV folds",
        "speaker_disjoint": "Unavailable: word-index filenames; no verified speaker metadata",
        "selection": "Highest training CV mean; ties use hamming before hann, SVM before RF",
        "best_model": best_name,
        "experiments": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "numpy",
                "scipy",
                "librosa",
                "soundfile",
                "scikit-learn",
                "joblib",
                "matplotlib",
            )
        },
    }
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
