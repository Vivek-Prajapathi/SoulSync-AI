from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"


def resolve_path(env_var: str, default: Path) -> Path:
    configured = os.getenv(env_var)
    if configured is None:
        return default
    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


MODEL_PATH = resolve_path("SOULSYNC_MODEL_PATH", BASE_DIR / "pose_cls_xgboost.joblib")
TFLITE_PATH = resolve_path("SOULSYNC_TFLITE_PATH", BASE_DIR / "movenet_thunder.tflite")
MAX_IMAGE_PIXELS = int(os.getenv("SOULSYNC_MAX_IMAGE_PIXELS", "1280000"))
MAX_IMAGE_BYTES = int(os.getenv("SOULSYNC_MAX_IMAGE_BYTES", "4194304"))
