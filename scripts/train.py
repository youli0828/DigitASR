import argparse
import json
from dataclasses import asdict
from pathlib import Path

from digit_asr.dataset import dataset_fingerprint, extract_split, read_manifest
from digit_asr.features import FeatureConfig
from digit_asr.models import create_model, save_model


def main():
    parser = argparse.ArgumentParser(description="Train using only the manifest train split")
    parser.add_argument("--manifest", type=Path, default=Path("data/samples/manifest.csv"))
    parser.add_argument(
        "--model", choices=["svm", "svm_hamming", "random_forest", "cnn"], default="svm"
    )
    parser.add_argument("--window", choices=["hann", "hamming"], default="hamming")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = FeatureConfig(
        window="hamming" if args.model == "svm_hamming" else args.window,
        sequence_frames=64 if args.model == "cnn" else None,
    )
    rows = read_manifest(args.manifest)
    x_train, y_train, selected = extract_split(rows, "train", config)
    model = create_model(args.model, args.seed)
    model.fit(x_train, y_train)
    metadata = {
        "model": args.model,
        "seed": args.seed,
        "features": asdict(config),
        "train_count": len(y_train),
        "train_paths": [r["path"] for r in selected],
        "dataset_sha256": dataset_fingerprint(rows),
    }
    name = "svm" if args.model == "svm_hamming" else args.model
    output = args.output or Path(f"artifacts/{name}_{config.window}/model.joblib")
    save_model(output, model, config, metadata)
    print(json.dumps({"saved": str(output), "train_count": len(y_train)}))


if __name__ == "__main__":
    main()
