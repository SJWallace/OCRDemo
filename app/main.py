# app/main.py
from __future__ import annotations

import io
import mimetypes
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter

import numpy as np
import cv2

from .ocr_engines import EasyOcrEngine, create_engines
from .trocr_engine import TrOCREngine


app = FastAPI(title="Handwriting OCR (EasyOCR + TrOCR)")


# Instantiate engines once
easy_engine = EasyOcrEngine(languages=["en"])
engine_registry = create_engines()  # {"easyocr": ..., "trocr": ...}
trocr_engine: TrOCREngine = engine_registry["trocr"]  # for advanced endpoint


# ---------- Static frontend ----------

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def frontend_root():
    return FileResponse("app/static/index.html")


# ---------- Preprocessing helpers ----------

def normalize_contrast(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    return img


def upscale(img: Image.Image, min_height: int = 1200) -> Image.Image:
    w, h = img.size
    if h >= min_height:
        return img
    scale = min_height / h
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)


def deskew(img: Image.Image) -> Image.Image:
    gray = np.array(img.convert("L"))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(np.array(img), M, (w, h), flags=cv2.INTER_CUBIC)
    return Image.fromarray(rotated)


def soften_background(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.MedianFilter(size=3))


def preprocess_page(img: Image.Image, quality: str = "fast") -> Image.Image:
    img = normalize_contrast(img)
    if quality == "high":
        img = deskew(img)
        img = upscale(img, min_height=1500)
        img = soften_background(img)
    return img


# ---------- File → pages helper ----------

def file_to_pages(file_bytes: bytes, filename: str) -> List[Image.Image]:
    mime, _ = mimetypes.guess_type(filename)
    if mime == "application/pdf" or filename.lower().endswith(".pdf"):
        try:
            pages = convert_from_bytes(file_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")
        return pages
    else:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.load()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        return [img]


# ---------- API ----------

@app.get("/api/models")
async def get_models():
    return {"models": list(engine_registry.keys())}


@app.get("/api/health")
async def health():
    return {"status": "ok", "models": list(engine_registry.keys())}


@app.post("/api/ocr")
async def run_ocr_simple(
    file: UploadFile = File(...),
    max_pages: int = Query(default=5, ge=1, le=100),
    quality: str = Query(
        default="fast",
        regex="^(fast|high)$",
        description="Preprocessing quality: 'fast' or 'high'",
    ),
):
    """
    Simple OCR endpoint: preprocessing + EasyOCR full-page.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    pages = file_to_pages(file_bytes, file.filename)
    if not pages:
        raise HTTPException(status_code=400, detail="No pages found in file")

    pages = pages[:max_pages]
    processed_pages = [preprocess_page(p, quality=quality) for p in pages]

    try:
        page_results = easy_engine.ocr_pages(processed_pages)
    except Exception as e:
        # Surface the real error instead of opaque 500
        raise HTTPException(
            status_code=500,
            detail=f"OCR failed for EasyOCR: {e}",
        )

    return JSONResponse({
        "file_name": file.filename,
        "models": {"easyocr": page_results},
    })


@app.post("/api/ocr_advanced")
async def run_ocr_advanced(
    file: UploadFile = File(...),
    max_pages: int = Query(default=3, ge=1, le=50),
    quality: str = Query(
        default="high",
        regex="^(fast|high)$",
        description="Preprocessing quality: 'fast' or 'high' (high recommended)",
    ),
):
    """
    Advanced handwriting pipeline:
      - preprocessing
      - line segmentation
      - TrOCR handwritten model per line
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    pages = file_to_pages(file_bytes, file.filename)
    if not pages:
        raise HTTPException(status_code=400, detail="No pages found in file")

    pages = pages[:max_pages]
    processed_pages = [preprocess_page(p, quality=quality) for p in pages]

    try:
        trocr_pages = trocr_engine.ocr_pages(processed_pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed for TrOCR: {e}")

    # Flatten per-page lines into a nicer shape:
    # { "trocr": [ {page, lines: [{bbox, text}, ...]}, ... ] }
    return JSONResponse({"file_name": file.filename, "models": {"trocr": trocr_pages}})
