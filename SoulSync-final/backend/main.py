from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ALLOWED_ORIGINS, FRONTEND_DIR, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, MODEL_PATH, TFLITE_PATH
from .inference import PoseAnalyzer, decode_base64_image, prepare_frame_for_inference

logger = logging.getLogger(__name__)
app = FastAPI(title="SoulSync AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.analyzer = PoseAnalyzer(str(MODEL_PATH), str(TFLITE_PATH))

app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1, max_length=MAX_IMAGE_BYTES * 4 // 3 + 256)
    expected_pose: Optional[str] = Field(default=None, max_length=50)
    pose: Optional[str] = Field(default=None, max_length=50)


class BatchAnalyzeRequest(BaseModel):
    images: List[AnalyzeRequest]


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


async def _analyze_single_payload(payload: AnalyzeRequest) -> Dict[str, Any]:
    image_payload = payload.image
    expected_pose = payload.expected_pose or payload.pose
    if not image_payload.strip():
        raise HTTPException(status_code=400, detail="A base64-encoded image is required.")

    try:
        image = decode_base64_image(image_payload)
        if image.nbytes > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Decoded image is too large to process.")
        image = prepare_frame_for_inference(image, max_pixels=MAX_IMAGE_PIXELS)
        return await asyncio.to_thread(app.state.analyzer.analyze, image, expected_pose)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pose analysis failed")
        raise HTTPException(status_code=500, detail="Pose analysis failed. Check backend logs for details.") from exc


@app.post("/api/analyze")
async def analyze_image(payload: AnalyzeRequest) -> Dict[str, Any]:
    return await _analyze_single_payload(payload)


@app.post("/api/analyze/batch")
async def analyze_batch(payload: BatchAnalyzeRequest) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for item in payload.images:
        results.append(await _analyze_single_payload(item))

    return {"success": True, "results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
