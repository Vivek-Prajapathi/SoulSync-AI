from __future__ import annotations

from typing import List, Optional

import numpy as np
from live_coach_xgboost import generic_coach_cues

POSE_DISPLAY_NAMES = {
    "chair": "Utkatasana",
    "cobra": "Bhujangasana",
    "dog": "Adho Mukha Svanasana",
    "warrior": "Virabhadrasana",
    "tree": "Vrksasana",
    "triangle": "Trikonasana",
    "traingle": "Trikonasana",
    "shoulder_stand": "Sarvangasana",
    "shoudler_stand": "Sarvangasana",
    "no_pose": "No pose",
    "default": "Yoga pose",
}

POSE_LABELS = {
    "chair": "Chair",
    "cobra": "Cobra",
    "dog": "Downward Dog",
    "warrior": "Warrior",
    "tree": "Tree",
    "triangle": "Triangle",
    "traingle": "Triangle",
    "shoulder_stand": "Shoulder Stand",
    "shoudler_stand": "Shoulder Stand",
    "no_pose": "No Pose",
}

KEYPOINT_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def normalize_pose_name(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "shoulder_stand": "shoudler_stand",
        "triangle": "traingle",
    }
    return aliases.get(value, value)


def display_name_for(label: str | None) -> str:
    if not label:
        return "Yoga pose"
    normalized = normalize_pose_name(label)
    return POSE_DISPLAY_NAMES.get(normalized, normalized.replace("_", " ").title())


def _joint_angle(
    keypoints: np.ndarray,
    first: str,
    middle: str,
    last: str,
    image_width: int,
    image_height: int,
) -> Optional[float]:
    indices = [KEYPOINT_INDEX[first], KEYPOINT_INDEX[middle], KEYPOINT_INDEX[last]]
    if any(keypoints[index][2] < 0.3 for index in indices):
        return None

    points = np.array(
        [
            [keypoints[index][1] * image_width, keypoints[index][0] * image_height]
            for index in indices
        ],
        dtype=np.float32,
    )
    first_vector = points[0] - points[1]
    second_vector = points[2] - points[1]
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator <= 1e-6:
        return None
    cosine = np.clip(np.dot(first_vector, second_vector) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def generate_feedback(
    keypoints: np.ndarray,
    predicted_label: str | None,
    expected_pose: str | None = None,
    image_width: int = 1,
    image_height: int = 1,
) -> List[str]:
    cues = list(generic_coach_cues(keypoints, kp_thresh=0.3))
    normalized_prediction = normalize_pose_name(predicted_label)
    normalized_expected = normalize_pose_name(expected_pose)

    if normalized_prediction == "no_pose":
        cues.insert(0, "Find your center and settle into a balanced stance.")
    elif normalized_prediction in {"chair", "warrior"}:
        knee_angles = [
            _joint_angle(keypoints, f"{side}_hip", f"{side}_knee", f"{side}_ankle", image_width, image_height)
            for side in ("left", "right")
        ]
        visible_angles = [angle for angle in knee_angles if angle is not None]
        if visible_angles and min(visible_angles) > 145:
            cues.insert(0, "Bend your knees a little more while keeping them aligned with your feet.")
        elif len(visible_angles) == 2 and abs(visible_angles[0] - visible_angles[1]) > 25:
            cues.insert(0, "Balance the bend between both legs for a steadier stance.")
    elif normalized_prediction == "tree":
        knee_angles = [
            _joint_angle(keypoints, f"{side}_hip", f"{side}_knee", f"{side}_ankle", image_width, image_height)
            for side in ("left", "right")
        ]
        if any(angle is not None and 145 <= angle < 165 for angle in knee_angles):
            cues.insert(0, "Lengthen your standing leg without locking the knee.")
    elif normalized_prediction == "cobra":
        elbow_angles = [
            _joint_angle(keypoints, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", image_width, image_height)
            for side in ("left", "right")
        ]
        if any(angle is not None and angle < 145 for angle in elbow_angles):
            cues.insert(0, "Ease the backbend by softening your elbows and relaxing your shoulders.")

    if expected_pose and normalized_prediction and normalized_prediction != normalized_expected:
        cues.insert(0, f"Move toward your selected {POSE_LABELS.get(normalized_expected, display_name_for(expected_pose))} pose.")
    if not cues:
        pose_name = POSE_LABELS.get(normalized_prediction, "pose")
        cues.append(f"Your {pose_name} landmarks are visible; keep breathing and move within a comfortable range.")

    if len(cues) > 3:
        return cues[:3]
    return cues
