from __future__ import annotations

import base64
import threading
import time
from typing import Any, Dict, Optional

import cv2
import joblib
import numpy as np
import pandas as pd

from live_coach_xgboost import MoveNetTFLite, decode_class_names, features_from_keypoints, topk_from_proba
from .coaching import display_name_for, generate_feedback, normalize_pose_name


def decode_base64_image(payload: str) -> np.ndarray:
    if not payload:
        raise ValueError("No image payload was provided.")

    content = payload.strip()
    if content.startswith("data:image"):
        header, _, encoded = content.partition(",")
        mime = header.replace("data:", "").replace(";base64", "")
        if mime and mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"Unsupported image type: {mime}")
        content = encoded

    try:
        raw = base64.b64decode(content, validate=True)
    except Exception:
        raw = base64.b64decode(content)

    buffer = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded image could not be decoded.")
    return image


def prepare_frame_for_inference(image: np.ndarray, max_pixels: int = 1_280_000) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Image is empty.")

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Image dimensions are invalid.")

    current_pixels = height * width
    if current_pixels > max_pixels:
        scale = max_pixels / current_pixels
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image


class PoseAnalyzer:
    def __init__(self, model_path: str, tflite_path: str) -> None:
        self._lock = threading.Lock()
        pack = joblib.load(model_path)
        self.model = pack["model"]
        self.feature_names = list(pack.get("features", []))
        self.class_names = list(decode_class_names(pack, self.model))
        self.use_scores = bool(pack.get("use_scores", False))
        self.movenet = MoveNetTFLite(tflite_path)

    def analyze(self, image: np.ndarray, expected_pose: Optional[str] = None) -> Dict[str, Any]:
        started_at = time.perf_counter()
        with self._lock:
            keypoints = self.movenet(image)
            features = features_from_keypoints(keypoints, self.feature_names, kp_min=-1.0, use_scores=self.use_scores)
            frame = pd.DataFrame([features], columns=self.feature_names)
            probabilities = self.model.predict_proba(frame)[0]

        top_predictions = topk_from_proba(probabilities, self.class_names, k=3)
        label, confidence = top_predictions[0]
        confidence_value = float(confidence)

        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        processed_keypoints = [
            {
                "name": name,
                "x": float(keypoints[index][1]),
                "y": float(keypoints[index][0]),
                "score": float(keypoints[index][2]),
            }
            for index, name in enumerate(keypoint_names)
        ]

        normalized_expected = normalize_pose_name(expected_pose)
        normalized_label = normalize_pose_name(label)
        is_correct_pose = True if not expected_pose else normalized_expected == normalized_label
        feedback = generate_feedback(keypoints, label, expected_pose)

        result = {
            "success": True,
            "pose": normalized_label,
            "display_name": display_name_for(label),
            "confidence": round(confidence_value, 4),
            "is_correct_pose": bool(is_correct_pose),
            "top_predictions": [
                {
                    "label": prediction_label,
                    "display_name": display_name_for(prediction_label),
                    "confidence": round(float(prediction_confidence), 4),
                }
                for prediction_label, prediction_confidence in top_predictions
            ],
            "keypoints": processed_keypoints,
            "feedback": feedback,
            "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
        return result
