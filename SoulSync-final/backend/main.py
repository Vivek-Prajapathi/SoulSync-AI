from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, MODEL_PATH, TFLITE_PATH
from .inference import PoseAnalyzer, decode_base64_image, prepare_frame_for_inference

app = FastAPI(title="SoulSync AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.analyzer = PoseAnalyzer(str(MODEL_PATH), str(TFLITE_PATH))

app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/index.html")
def read_index_html() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/session")
def read_session() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "session.html")


@app.get("/session.html")
def read_session_html() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "session.html")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "SoulSync AI",
        "model_path": str(MODEL_PATH),
        "tflite_path": str(TFLITE_PATH),
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "max_image_bytes": MAX_IMAGE_BYTES,
    }


async def _analyze_single_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    image_payload = payload.get("image")
    expected_pose = payload.get("expected_pose") or payload.get("pose")

    if not isinstance(image_payload, str) or not image_payload.strip():
        raise HTTPException(status_code=400, detail="A base64-encoded image is required.")

    image = decode_base64_image(image_payload)
    if image.nbytes > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large to process.")

    image = prepare_frame_for_inference(image, max_pixels=MAX_IMAGE_PIXELS)
    return await asyncio.to_thread(app.state.analyzer.analyze, image, expected_pose)


@app.post("/api/analyze")
async def analyze_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return await _analyze_single_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - server-side guard
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.post("/api/analyze/batch")
async def analyze_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    images = payload.get("images")
    if not isinstance(images, list):
        raise HTTPException(status_code=400, detail="A list of images is required under the 'images' field.")

    results: List[Dict[str, Any]] = []
    for item in images:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Each item in 'images' must be an object containing an image.")
        results.append(await _analyze_single_payload(item))

    return {"success": True, "results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
