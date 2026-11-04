"""Run all four fixed experiments and preserve measured reports."""

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/samples/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in ("svm", "random_forest"):
        for window in ("hamming", "hann"):
            name = f"{model}_{window}"
            artifact = root / "artifacts" / name / "model.joblib"
            report_dir = args.output / name
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/train.py"),
                    "--manifest",
                    str(args.manifest),
                    "--model",
                    model,
                    "--window",
                    window,
                    "--seed",
                    "42",
                    "--output",
                    str(artifact),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/eval.py"),
                    "--manifest",
                    str(args.manifest),
                    "--model",
                    str(artifact),
                    "--output",
                    str(report_dir),
                ],
                check=True,
            )
            report = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
            summaries.append(
                {
                    "model": model,
                    "window": window,
                    "train_count": report["train_count"],
                    "correct": report["correct"],
                    "test_count": report["total"],
                    "accuracy": report["accuracy"],
                    "dataset_sha256": report["dataset_sha256"],
                }
            )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "librosa",
                "soundfile",
                "scikit-learn",
                "joblib",
                "espeakng-loader",
                "gradio",
                "pytest",
                "ruff",
            )
        },
    }
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
