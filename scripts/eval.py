import argparse
import csv
import json
from pathlib import Path

from digit_asr.dataset import dataset_fingerprint, extract_split, read_manifest
from digit_asr.evaluate import evaluate_predictions, export_report
from digit_asr.models import load_model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the held-out test split"
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/samples/manifest.csv"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/svm_hamming/model.joblib"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = load_model(args.model)
    rows = read_manifest(args.manifest)
    if dataset_fingerprint(rows) != bundle["metadata"]["dataset_sha256"]:
        raise ValueError("Dataset differs from the training manifest/audio snapshot")
    x_test, y_test, selected = extract_split(rows, "test", bundle["config"])
    predictions = bundle["estimator"].predict(x_test)
    report = evaluate_predictions(y_test, predictions)
    report.update(bundle["metadata"])
    report["test_paths"] = [row["path"] for row in selected]
    output = args.output or args.model.parent
    export_report(report, output)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "prediction"])
        for row, prediction in zip(selected, predictions):
            writer.writerow([row["path"], row["label"], int(prediction)])
    print(json.dumps({k: report[k] for k in ("model", "accuracy", "correct", "total")}))


if __name__ == "__main__":
    main()
