import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_predictions(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("Expected nonempty matching one-dimensional labels")
    if not np.isin(y_true, range(10)).all() or not np.isin(y_pred, range(10)).all():
        raise ValueError("Labels must be digits 0-9")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "correct": int(np.sum(y_true == y_pred)),
        "total": int(y_true.size),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(10)), output_dict=True, zero_division=0
        ),
    }


def export_report(report: dict, output: str | Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *range(10)])
        for digit, row in enumerate(report["confusion_matrix"]):
            writer.writerow([digit, *row])
