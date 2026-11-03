# SoulSync AI

This repository contains the SoulSync AI yoga pose detection and coaching project. The implementation lives under the [SoulSync-final](./SoulSync-final) folder and runs live pose inference in the browser.

## Project Summary

SoulSync AI detects yoga poses using the existing MoveNet TFLite model and trained XGBoost classifier. Camera frames stay in the browser; the FastAPI backend is optional.

## Quick Start

```bash
cd SoulSync-final/frontend
npm ci
npm run dev
```

Then open:

- http://localhost:5173/
- http://localhost:5173/session.html

## Project Folder

- [SoulSync-final/backend](./SoulSync-final/backend) — optional FastAPI backend and pose inference logic
- [SoulSync-final/frontend](./SoulSync-final/frontend) — Vite frontend, browser inference, and session UI
- [SoulSync-final/frontend/models](./SoulSync-final/frontend/models) — parity-validated ONNX classifier and metadata
- [SoulSync-final/live_coach_xgboost.py](./SoulSync-final/live_coach_xgboost.py) — MoveNet + XGBoost live coaching pipeline
- [SoulSync-final/pose_cls_xgboost.joblib](./SoulSync-final/pose_cls_xgboost.joblib) — trained XGBoost model bundle
- [SoulSync-final/movenet_thunder.tflite](./SoulSync-final/movenet_thunder.tflite) — MoveNet TFLite model

## Deployment

Deploy the static app to Vercel with `SoulSync-final` as the project root. Its `vercel.json` builds and publishes the frontend and browser model assets; no separately hosted API is needed.

- Local demo: http://127.0.0.1:8000
- Production URL: replace this with your live deployment URL

## Documentation

The detailed project documentation is available in [SoulSync-final/README.md](./SoulSync-final/README.md).
