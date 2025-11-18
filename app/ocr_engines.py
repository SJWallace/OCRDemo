# app/ocr_engines.py
from __future__ import annotations

from typing import List, Dict, Any

from PIL import Image
import numpy as np
import easyocr

from .trocr_engine import TrOCREngine


class OcrEngine:
    name: str

    def ocr_pages(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class EasyOcrEngine(OcrEngine):
    """
    Page-level EasyOCR, tuned for handwriting-ish docs.
    """

    def __init__(self, languages: list[str] | None = None):
        self.name = "easyocr"
        self._reader = easyocr.Reader(languages or ["en"])

    def ocr_pages(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for idx, img in enumerate(images, start=1):
            img_arr = np.array(img)
            ocr_result = self._reader.readtext(
                img_arr,
                detail=1,
                decoder="beamsearch",
                contrast_ths=0.05,
                adjust_contrast=0.7,
                paragraph=False,
                allowlist=(
                    "0123456789"
                    "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    ".,!?;:'\"-()[]/\\@&%$ "
                ),
            )
            lines = [text for (_bbox, text, _conf) in ocr_result]
            page_text = "\n".join(lines)
            results.append({"page": idx, "text": page_text})

        return results


# Expose TrOCR via this module too
def create_engines():
    easy = EasyOcrEngine(languages=["en"])
    trocr = TrOCREngine()
    return {
        easy.name: easy,
        trocr.name: trocr,
    }
