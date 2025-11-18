from __future__ import annotations

from typing import List, Dict, Any

import torch
from PIL import Image

from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from .line_segmenter import LineImage, segment_lines


class TrOCREngine:
    """
    Handwriting-optimised recogniser using Microsoft's TrOCR model.
    Works on a list of preprocessed page images via line segmentation.
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        self.name = "trocr"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _recognize_line(self, img: Image.Image) -> str:
        # Ensure 3-channel RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Single-image path; let HF handle resizing etc.
        encoding = self.processor(
            images=img,
            return_tensors="pt",
        )
        pixel_values = encoding.pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        return text.strip()

    def ocr_pages(self, pages: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Returns a list of page dicts:
          [{ "page": 1, "lines": [ { "bbox": .., "text": "..." }, ... ] }, ...]
        """
        out: List[Dict[str, Any]] = []

        for page_index, page in enumerate(pages, start=1):
            line_imgs: List[LineImage] = segment_lines(page)
            line_results: List[Dict[str, Any]] = []

            for line in line_imgs:
                text = self._recognize_line(line.image)
                if not text:
                    continue
                x1, y1, x2, y2 = line.bbox
                line_results.append({
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "text": text,
                })

            out.append({
                "page": page_index,
                "lines": line_results,
            })

        return out
