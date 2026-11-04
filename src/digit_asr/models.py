from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from digit_asr.features import FeatureConfig


def create_model(name: str, seed: int = 42):
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", gamma="scale"))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    raise ValueError(f"Unknown model: {name}")


def save_model(path: str | Path, estimator, config: FeatureConfig, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": estimator, "config": config, "metadata": metadata}, path)


def load_model(path: str | Path) -> dict:
    """Load a trusted local artifact; joblib files can execute Python code."""
    return joblib.load(path)
