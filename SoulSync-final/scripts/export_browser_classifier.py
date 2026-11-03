from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "pose_cls_xgboost.joblib"
DEFAULT_TEST_CSV = PROJECT_ROOT / "test_data_new.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "frontend" / "models" / "pose_cls_xgboost.onnx"
DEFAULT_TOLERANCE = 1e-5


def export_classifier(
    model_path: Path,
    test_csv_path: Path,
    output_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        pack = joblib.load(model_path)

    model = pack["model"]
    features = list(pack["features"])
    labels = [str(label) for label in pack["labels"].classes_]
    if len(features) != 51 or len(labels) != 8:
        raise ValueError(
            f"Expected the trained 51-feature, 8-class model; found "
            f"{len(features)} features and {len(labels)} classes."
        )
    if not pack.get("use_scores", False):
        raise ValueError("The saved classifier does not declare use_scores=True.")

    try:
        estimator = model.named_steps["xgb"]
    except (AttributeError, KeyError) as exc:
        raise ValueError("Expected the saved model to contain an 'xgb' pipeline step.") from exc
    if not isinstance(estimator, XGBClassifier):
        raise TypeError(f"Expected XGBClassifier, found {type(estimator).__name__}.")

    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )
    converted = convert_sklearn(
        model,
        "soulsync_pose_classifier",
        [("input", FloatTensorType([None, len(features)]))],
        target_opset={"": 17, "ai.onnx.ml": 3},
        options={id(estimator): {"zipmap": False}},
    )
    onnx.checker.check_model(converted)

    test_frame = pd.read_csv(test_csv_path)
    missing_features = [name for name in features if name not in test_frame.columns]
    if missing_features:
        raise ValueError(f"Test CSV is missing model features: {missing_features}")

    test_features = test_frame[features]
    python_probabilities = model.predict_proba(test_features)
    python_class_indices = np.asarray(model.classes_)[np.argmax(python_probabilities, axis=1)]

    session = ort.InferenceSession(
        converted.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    label_output, probability_output = session.run(
        None,
        {input_name: test_features.to_numpy(dtype=np.float32)},
    )
    browser_class_indices = np.asarray(label_output).reshape(-1).astype(np.int64)
    browser_probabilities = np.asarray(probability_output, dtype=np.float64)

    if browser_probabilities.shape != python_probabilities.shape:
        raise ValueError(
            f"ONNX probability shape {browser_probabilities.shape} does not match "
            f"Python shape {python_probabilities.shape}."
        )
    class_mismatches = int(np.count_nonzero(browser_class_indices != python_class_indices))
    max_probability_error = float(np.max(np.abs(browser_probabilities - python_probabilities)))
    mean_probability_error = float(np.mean(np.abs(browser_probabilities - python_probabilities)))
    if class_mismatches:
        raise ValueError(
            f"ONNX predictions differ from Python on {class_mismatches}/{len(test_frame)} test rows."
        )
    if max_probability_error > tolerance:
        raise ValueError(
            f"ONNX probabilities differ from Python by up to {max_probability_error:.8g}; "
            f"allowed tolerance is {tolerance:.8g}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converted.SerializeToString())
    metadata = {
        "format_version": 1,
        "feature_count": len(features),
        "feature_order": features,
        "use_scores": bool(pack["use_scores"]),
        "class_labels": labels,
        "class_indices": [int(value) for value in model.classes_],
        "input_name": input_name,
        "outputs": {
            "label": session.get_outputs()[0].name,
            "probabilities": session.get_outputs()[1].name,
        },
        "parity": {
            "test_rows": int(len(test_frame)),
            "class_mismatches": class_mismatches,
            "max_probability_absolute_error": max_probability_error,
            "mean_probability_absolute_error": mean_probability_error,
            "tolerance": tolerance,
        },
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the existing XGBoost pipeline to ONNX after test-set parity checks."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative.")
    metadata = export_classifier(args.model, args.test_csv, args.output, args.tolerance)
    print(
        "ONNX parity passed: "
        f"{metadata['parity']['test_rows']} rows, "
        f"{metadata['parity']['class_mismatches']} class mismatches, "
        f"max probability error "
        f"{metadata['parity']['max_probability_absolute_error']:.8g}."
    )
    print(f"ONNX model: {args.output}")
    print(f"Metadata: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
