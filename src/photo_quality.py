"""PMS 原圖與下載內容共用的可用性檢查。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError

MIN_PHOTO_BYTES = 1024

# PMS 曾有可救回的截斷 JPEG；與 prepare.py 的實際解碼政策一致。
ImageFile.LOAD_TRUNCATED_IMAGES = True


def bytes_problem(blob: bytes) -> str | None:
    if len(blob) < MIN_PHOTO_BYTES:
        return f"too_small:{len(blob)}"
    try:
        with Image.open(BytesIO(blob)) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                return "invalid_dimensions"
    except (OSError, ValueError, UnidentifiedImageError):
        return "decode_error"
    return None


def file_problem(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    if path.stat().st_size < MIN_PHOTO_BYTES:
        return f"too_small:{path.stat().st_size}"
    try:
        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                return "invalid_dimensions"
    except (OSError, ValueError, UnidentifiedImageError):
        return "decode_error"
    return None
