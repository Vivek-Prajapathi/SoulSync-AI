# SoulSync AI

This repository contains the SoulSync AI yoga pose detection and coaching project. The implementation lives under the [SoulSync-final](./SoulSync-final) folder.

## Project Summary

SoulSync AI detects yoga poses from camera frames using MoveNet keypoints and a trained XGBoost classifier. The application serves a browser-based practice flow with live analysis, posture coaching, and a FastAPI backend for inference.

## Quick Start

```bash
cd SoulSync-final
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then open:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/session

## Project Folder

- [SoulSync-final/backend](./SoulSync-final/backend) — FastAPI backend and pose inference logic
- [SoulSync-final/frontend](./SoulSync-final/frontend) — landing page and session UI
- [SoulSync-final/live_coach_xgboost.py](./SoulSync-final/live_coach_xgboost.py) — MoveNet + XGBoost live coaching pipeline
- [SoulSync-final/pose_cls_xgboost.joblib](./SoulSync-final/pose_cls_xgboost.joblib) — trained XGBoost model bundle
- [SoulSync-final/movenet_thunder.tflite](./SoulSync-final/movenet_thunder.tflite) — MoveNet TFLite model

## Deployment

The app is designed to run as a Python web service on a CPU-friendly deployment target. For local testing, use the backend instructions above.

- Local demo: http://127.0.0.1:8000
- Production URL: replace this with your live deployment URL

## Documentation

The detailed project documentation is available in [SoulSync-final/README.md](./SoulSync-final/README.md).
