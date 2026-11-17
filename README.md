# SoulSync AI

SoulSync AI is a yoga posture detection and coaching system that runs MoveNet keypoint extraction and the trained XGBoost classifier directly in the browser. Users can choose a pose, frame themselves on camera, and receive live posture feedback without sending camera frames to an analysis server.

## Live Demo

- Local development: http://127.0.0.1:8000
- Production deployment: replace with your deployed URL

## Features

- Webcam-based live practice flow
- Browser-side MoveNet TFLite inference for 17 body keypoints
- Browser-side ONNX export of the original XGBoost pipeline
- Confidence-coloured keypoint and skeleton overlay on the mirrored live preview
- Top-3 prediction output with confidence scores
- Live keypoint visibility and pose-detection status
- Coaching feedback tuned to posture quality and target alignment
- Optional FastAPI backend retained for API-based integration and testing
- Browser-based session selection and camera preview

## Project Structure

```text
SoulSync-final/
├── backend/
│   ├── __init__.py
│   ├── .env.example
│   ├── config.py
│   ├── coaching.py
│   ├── inference.py
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── css/
│   ├── js/
│   ├── models/
│   │   ├── pose_cls_xgboost.json
│   │   └── pose_cls_xgboost.onnx
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   ├── index.html
│   └── session.html
├── scripts/
│   └── export_browser_classifier.py
├── browser-conversion-requirements.txt
├── vercel.json
├── live_coach_xgboost.py
├── movenet.py
├── classify_with_xgboost.py
├── create_movenet_csv.py
├── train_xgboost.py
├── train_rf_from_csv.py
├── eval_on_test_csv.py
├── preprocessing.py
├── data.py
├── training.py
├── movenet_thunder.tflite
├── pose_cls_xgboost.joblib
├── pose_cls_movenet_new.joblib
├── pose_cls_movenet_with_scores.joblib
├── weights.best.hdf5
├── train_data_new.csv
├── test_data_new.csv
├── predictions.csv
├── final_xgboost_predictions.csv
├── venv/
└── README.md
```

## Models

The project uses the trained XGBoost bundle saved as `pose_cls_xgboost.joblib` and the MoveNet TFLite weights in `movenet_thunder.tflite`.

`movenet_thunder.tflite` runs in the browser through the TensorFlow.js TFLite WebAssembly runtime. `pose_cls_xgboost.joblib` remains the source model; `scripts/export_browser_classifier.py` exports its complete `StandardScaler` + `XGBClassifier` pipeline to ONNX. The browser artifact retains the saved 51-feature order, `use_scores=True`, and original class-label mapping.

The export script compares ONNX predictions with the Python pipeline on every row in `test_data_new.csv` before writing the browser artifact. The current export has 0 class mismatches across 1,011 rows; its maximum probability absolute difference is 3.43e-7 (the allowed tolerance is 1e-5). The original `.joblib` and `.tflite` models are not modified.

The existing model may emit scikit-learn/XGBoost version warnings when loaded by the current Python environment. The browser export validation uses the installed Python model as its reference and does not retrain it.

## Supported Pose Labels

The classifier outputs the following labels from the original model bundle:

- chair -> Utkatasana
- cobra -> Bhujangasana
- dog -> Adho Mukha Svanasana
- no_pose -> No pose
- shoudler_stand -> Sarvangasana
- traingle -> Trikonasana
- tree -> Vrksasana
- warrior -> Virabhadrasana

## Backend API

### Health

- GET `/health`

Example:

```bash
curl http://127.0.0.1:8000/health
```

### Analyze a submitted frame

- POST `/api/analyze`

Request body example:

```json
{
  "image": "data:image/jpeg;base64,...",
  "expected_pose": "cobra",
  "pose": "cobra"
}
```

Response fields include `pose`, `display_name`, `confidence`, `confidence_type`, `confidence_calibrated`, `person_detected`, `pose_detection_status`, `keypoint_confidence`, `visible_keypoints`, `top_predictions`, `keypoints`, `feedback`, and `processing_time_ms`.

When a person is detected, `confidence` and each top prediction score come from the trained model's `predict_proba` output. These are prediction scores, not measured model accuracy, and are not calibrated probabilities. `keypoints` always contains all 17 MoveNet landmarks with source-frame normalized coordinates and MoveNet confidence scores. Frames without a detected person do not receive a classifier prediction.

### Batch analysis

- POST `/api/analyze/batch`

This endpoint accepts a list of analysis payloads for bulk usage.

## Run the Browser App Locally

The frontend performs live inference locally and does not require the FastAPI server.

```bash
cd SoulSync-final/frontend
npm ci
npm run dev
```

Open `http://localhost:5173/` and start a session. Camera access requires `localhost` or HTTPS. The first run downloads the 12.6 MB MoveNet model and WebAssembly runtime assets; later visits can use the browser cache.

To validate the production build locally:

```bash
cd SoulSync-final/frontend
npm run build
npm run preview
```

## Regenerate and Validate the ONNX Classifier

The ONNX classifier artifact is generated from the existing Python model. To regenerate it, first install the backend's Python requirements, then install the conversion-only tools and run the exporter:

```bash
cd SoulSync-final
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
pip install -r browser-conversion-requirements.txt
python scripts/export_browser_classifier.py
```

The exporter stops without replacing the existing browser artifact if any class differs from Python or if a probability differs by more than `1e-5`.

## Vercel Deployment

Set the Vercel project root directory to `SoulSync-final`. The checked-in `vercel.json` runs `npm --prefix frontend ci`, builds the Vite site, and publishes `frontend/dist`. The build includes the original MoveNet TFLite asset, the validated ONNX classifier, and the required WebAssembly files. No FastAPI host or `/api/analyze` route is required by the deployed frontend.

## Deployment Notes

Vercel serves only static assets for this setup. Browser inference uses WebAssembly on the visitor's device; actual frame rate depends on the device and browser. The TensorFlow.js TFLite package is alpha software, and model inference should be checked in the target browsers after deployment.

The optional FastAPI routes remain available for other clients and direct API testing, but the session page does not call them.

## Troubleshooting

### Optional FastAPI service

The backend remains available for clients that explicitly use its API. Install its Python requirements and start it with:

```bash
cd SoulSync-final
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Model compatibility warnings

The model may emit scikit-learn or XGBoost version warnings when loaded in a newer environment. This is expected when the artifact was created with an older release. The project preserves the original model and avoids retraining unless explicitly required.

### Camera not working in browser

Use:

- a local HTTPS origin or localhost
- a browser with webcam permissions enabled
- a supported webcam device

## Development Notes

The browser feature vector is built in the original 51-feature order recorded in the model bundle metadata. The ONNX graph includes the original scaler and classifier; JavaScript does not rescale or reorder those features.

## License

This project is intended for educational and personal use. Update the license before production deployment if you plan to distribute it externally.
