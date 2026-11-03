from __future__ import annotations

from typing import List

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


def normalize_pose_name(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return value


def display_name_for(label: str | None) -> str:
    if not label:
        return "Yoga pose"
    normalized = normalize_pose_name(label)
    return POSE_DISPLAY_NAMES.get(normalized, normalized.replace("_", " ").title())


def generate_feedback(keypoints, predicted_label: str | None, expected_pose: str | None = None) -> List[str]:
    cues = list(generic_coach_cues(keypoints, kp_thresh=0.3))
    normalized_prediction = normalize_pose_name(predicted_label)
    normalized_expected = normalize_pose_name(expected_pose)

    if expected_pose and normalized_prediction and normalized_prediction != normalized_expected:
        cues.insert(0, f"Adjust into {display_name_for(expected_pose)} alignment.")
    elif normalized_prediction == "no_pose":
        cues.insert(0, "Find your center and settle into a balanced stance.")
    elif not cues:
        cues.append("Keep your spine long and your gaze steady.")

    if len(cues) > 3:
        return cues[:3]
    return cues
