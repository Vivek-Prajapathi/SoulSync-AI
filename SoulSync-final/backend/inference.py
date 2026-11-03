from __future__ import annotations

import base64
import binascii
import threading
import time
from typing import Any, Dict, Optional

import cv2
import joblib
import numpy as np
import pandas as pd

from live_coach_xgboost import KEYPOINT_NAMES, MoveNetTFLite, decode_class_names, features_from_keypoints, topk_from_proba
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
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image data must be valid base64.") from exc

    buffer = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded image could not be decoded.")
    return image


def keypoints_to_image_coordinates(keypoints: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Convert MoveNet's padded-square coordinates into source-image coordinates."""
    coordinates = np.asarray(keypoints, dtype=np.float32).copy()
    if image_width > image_height:
        scale = image_height / image_width
        vertical_padding = (1.0 - scale) / 2.0
        coordinates[:, 0] = (coordinates[:, 0] - vertical_padding) / scale
    elif image_height > image_width:
        scale = image_width / image_height
        horizontal_padding = (1.0 - scale) / 2.0
        coordinates[:, 1] = (coordinates[:, 1] - horizontal_padding) / scale
    return coordinates


def detect_person(keypoints: np.ndarray) -> bool:
    visible = np.asarray(keypoints)[:, 2] >= 0.3
    shoulder_visible = bool(visible[5] or visible[6])
    hip_visible = bool(visible[11] or visible[12])
    return int(np.count_nonzero(visible)) >= 6 and shoulder_visible and hip_visible


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
        image_height, image_width = image.shape[:2]
        with self._lock:
            keypoints = self.movenet(image)
            found_person = detect_person(keypoints)
            probabilities = None
            if found_person:
                features = features_from_keypoints(
                    keypoints,
                    self.feature_names,
                    kp_min=-1.0,
                    use_scores=self.use_scores,
                )
                frame = pd.DataFrame([features], columns=self.feature_names)
                probabilities = self.model.predict_proba(frame)[0]

        keypoints_for_display = keypoints_to_image_coordinates(keypoints, image_width, image_height)
        processed_keypoints = [
            {
                "name": name,
                "x": float(keypoints_for_display[index][1]),
                "y": float(keypoints_for_display[index][0]),
                "score": float(keypoints[index][2]),
            }
            for index, name in enumerate(KEYPOINT_NAMES)
        ]

        keypoint_confidence = float(np.mean(keypoints[:, 2]))
        visible_keypoints = int(np.count_nonzero(keypoints[:, 2] >= 0.3))
        pose_detection_status = (
            "Good"
            if found_person and visible_keypoints >= 12 and keypoint_confidence >= 0.5
            else "Partial"
            if found_person
            else "Not detected"
        )

        if found_person and probabilities is not None:
            top_predictions = topk_from_proba(probabilities, self.class_names, k=3)
            label, confidence = top_predictions[0]
            confidence_value: Optional[float] = float(confidence)
            prediction_list = [
                {
                    "label": prediction_label,
                    "display_name": display_name_for(prediction_label),
                    "confidence": round(float(prediction_confidence), 4),
                }
                for prediction_label, prediction_confidence in top_predictions
            ]
            normalized_expected = normalize_pose_name(expected_pose)
            normalized_label = normalize_pose_name(label)
            is_correct_pose = not expected_pose or normalized_expected == normalized_label
            feedback = generate_feedback(
                keypoints_for_display,
                label,
                expected_pose,
                image_width=image_width,
                image_height=image_height,
            )
            pose = normalized_label
            display_name = display_name_for(label)
        else:
            confidence_value = None
            prediction_list = []
            is_correct_pose = False if expected_pose else None
            feedback = ["Please move into the camera frame so your body can be detected clearly."]
            pose = "no_pose"
            display_name = "No pose detected"

        result = {
            "success": True,
            "pose": pose,
            "display_name": display_name,
            "confidence": round(confidence_value, 4) if confidence_value is not None else None,
            "confidence_type": "predict_proba" if confidence_value is not None else None,
            "confidence_calibrated": False if confidence_value is not None else None,
            "is_correct_pose": is_correct_pose,
            "person_detected": found_person,
            "pose_detection_status": pose_detection_status,
            "keypoint_confidence": round(keypoint_confidence, 4),
            "visible_keypoints": visible_keypoints,
            "top_predictions": prediction_list,
            "keypoints": processed_keypoints,
            "feedback": feedback,
            "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
        return result
