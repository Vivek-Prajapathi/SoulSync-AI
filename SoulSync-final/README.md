# SoulSync AI

SoulSync AI is a yoga posture detection and coaching system that uses MoveNet for keypoint extraction and a trained XGBoost classifier for pose recognition. It combines computer vision, motion analysis, and a lightweight web frontend so users can choose a pose, frame themselves on camera, and receive posture feedback in near real time.

## Live Demo

- Local development: http://127.0.0.1:8000
- Production deployment: replace with your deployed URL

## Features

- Webcam-based live practice flow
- MoveNet TFLite inference for 17 body keypoints
- XGBoost pose classification using the trained model bundle
- Confidence-coloured keypoint and skeleton overlay on the mirrored live preview
- Top-3 prediction output with confidence scores
- Live keypoint visibility and pose-detection status
- Coaching feedback tuned to posture quality and target alignment
- FastAPI backend with health checks and structured JSON responses
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
│   ├── index.html
│   └── session.html
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

Important compatibility note:
- The serialized model was created with older scikit-learn and XGBoost versions.
- The project is configured to keep the original model and use a matching runtime environment where possible.
- The current environment uses Python 3.11 and retains the compatible package versions already present in the project virtual environment.

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

## Local Setup

### 1) Open the project folder

```bash
cd SoulSync-final
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 4) Run the backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 5) Open the app

```text
http://127.0.0.1:8000/
```

## Frontend Usage

The frontend pages are served by FastAPI and can be accessed through the backend.

- Home page: `/`
- Session page: `/session`

The user selects a pose and turns on the camera. The browser sends reduced JPEG frames to the backend at up to about one frame per second. A confidence-coloured skeleton is drawn over the mirrored camera preview and the dashboard displays the detected pose, model score, keypoint confidence, status, practice time, and coaching cues. The app processes uploaded frames in memory and does not save image frames.

## Deployment Notes

This project is suitable for CPU-based deployment with a lightweight backend workload.

Recommended deployment principles:

- keep the model files in the app bundle or mounted volume
- use a single FastAPI worker process with model initialization at startup
- enforce reasonable maximum image size limits
- avoid loading or reloading the model on every request
- run behind a reverse proxy with HTTPS enabled
- when the frontend and API are deployed separately, set `SOULSYNC_ALLOWED_ORIGINS` on the backend to the exact frontend origin
- set the `soulsync-api-base` meta tag in `frontend/session.html` to the HTTPS URL of the deployed FastAPI service

The Vercel frontend is static and does not provide FastAPI routes. If the backend URL is not configured, the session page will keep the camera preview available and show a setup message instead of repeatedly requesting a missing `/api/analyze` route. Live MoveNet/XGBoost predictions require the FastAPI backend to be deployed and reachable.

For separate frontend/backend deployments:

1. Deploy the FastAPI app with `backend/requirements.txt`, `pose_cls_xgboost.joblib`, and `movenet_thunder.tflite`.
2. Set `SOULSYNC_ALLOWED_ORIGINS` on the backend to the exact HTTPS frontend origin, for example `https://your-site.vercel.app`.
3. Set the `content` of `<meta name="soulsync-api-base">` in `frontend/session.html` to the FastAPI service origin, for example `https://your-api-host.example`.
4. Redeploy the frontend. The browser then posts frames to `<backend-origin>/api/analyze`.

## Troubleshooting

### FastAPI fails to start

Check that:

- the virtual environment is active
- dependencies were installed successfully
- `movenet_thunder.tflite` exists in the project root
- `pose_cls_xgboost.joblib` exists in the project root

### Model compatibility warnings

The model may emit scikit-learn or XGBoost version warnings when loaded in a newer environment. This is expected when the artifact was created with an older release. The project preserves the original model and avoids retraining unless explicitly required.

### Camera not working in browser

Use:

- a local HTTPS origin or localhost
- a browser with webcam permissions enabled
- a supported webcam device

## Development Notes

The app reuses the original feature construction and inference pipeline implemented in `live_coach_xgboost.py` as much as possible so the trained model input remains consistent.

## License

This project is intended for educational and personal use. Update the license before production deployment if you plan to distribute it externally.
