"""VLM API object detector using an OpenAI-compatible chat completion endpoint."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image


@dataclass
class Detection:
    found: bool
    bbox: tuple[int, int, int, int] | None
    confidence: float
    raw_text: str


class VLMDetector:
    """Detect target object from RGB image via VLM API."""

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        model: str = "gpt-4o-mini",
        timeout_s: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "VLMDetector | None":
        api_key = os.getenv("VLM_API_KEY", "").strip()
        if not api_key:
            return None
        api_url = os.getenv("VLM_API_URL", "https://api.openai.com/v1/chat/completions").strip()
        model = os.getenv("VLM_MODEL", "gpt-4o-mini").strip()
        timeout_s = float(os.getenv("VLM_TIMEOUT_S", "15"))
        return cls(api_key=api_key, api_url=api_url, model=model, timeout_s=timeout_s)

    def _image_to_data_url(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
        return None

    def detect(self, image: Image.Image, target: str) -> Detection:
        w, h = image.size
        prompt = (
            "You are an object detector. Return strict JSON only.\n"
            f"Detect target object '{target}' in this indoor image.\n"
            f"If found, return: {{\"found\": true, \"bbox\": [x1,y1,x2,y2], \"confidence\": 0.0-1.0}}.\n"
            f"If not found, return: {{\"found\": false, \"bbox\": null, \"confidence\": 0.0}}.\n"
            f"Image resolution is width={w}, height={h}. "
            "bbox must be integer pixel coordinates and clipped to image bounds."
        )
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                    ],
                }
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(self.api_url, headers=headers, json=body, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except Exception as exc:
            return Detection(found=False, bbox=None, confidence=0.0, raw_text=f"api_error:{exc}")

        parsed = self._extract_json(text) or {}
        found = bool(parsed.get("found", False))
        conf = float(parsed.get("confidence", 0.0) or 0.0)
        bbox_raw = parsed.get("bbox")
        bbox: tuple[int, int, int, int] | None = None
        if found and isinstance(bbox_raw, list) and len(bbox_raw) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox_raw]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 > x1 and y2 > y1:
                bbox = (x1, y1, x2, y2)
            else:
                found = False

        return Detection(found=found and bbox is not None, bbox=bbox, confidence=conf, raw_text=text)
