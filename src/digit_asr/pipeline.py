import argparse
import json
from pathlib import Path

from digit_asr.features import features_from_file
from digit_asr.models import load_model


class DigitRecognizer:
    def __init__(self, model_path: str | Path):
        bundle = load_model(model_path)
        self.estimator = bundle["estimator"]
        self.config = bundle["config"]
        self.metadata = bundle["metadata"]

    def predict(self, audio_path: str | Path) -> int:
        vector = features_from_file(audio_path, self.config)
        return int(self.estimator.predict(vector[None, :])[0])


def main():
    parser = argparse.ArgumentParser(description="Recognize one isolated Chinese spoken digit")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/svm_hamming/model.joblib"))
    args = parser.parse_args()
    print(json.dumps({"digit": DigitRecognizer(args.model).predict(args.audio)}))


if __name__ == "__main__":
    main()
