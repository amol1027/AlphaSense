"""Model artifact save/load for the frozen volatility model."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib

FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
    "range_mean_1h",
    "range_max_1h",
    "ret_std_1h",
]

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "volatility"


@dataclass
class VolatilityBundle:
    asset: str
    features: list
    scaler: object
    model: object
    range_median: float
    n_train: int
    trained_at: str


def artifact_path(asset: str) -> Path:
    return ARTIFACT_DIR / f"{asset.lower()}.joblib"


def save_bundle(bundle: VolatilityBundle) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = artifact_path(bundle.asset)
    joblib.dump(bundle, path)
    return path


@lru_cache(maxsize=5)
def load_bundle(asset: str) -> VolatilityBundle:
    path = artifact_path(asset)
    if not path.exists():
        raise ValueError(
            f"No trained artifact for asset {asset!r}. "
            "Run scripts/train_volatility_model.py first."
        )
    bundle = joblib.load(path)
    if list(bundle.features) != FEATURES:
        raise ValueError(f"Artifact feature mismatch for {asset!r}")
    return bundle
