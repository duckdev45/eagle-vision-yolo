"""Gemini 佐證框（anno.evidence → manifest.predBoxes）的共用工具。

座標是 0~1000 的相對值，乘上實際尺寸才是像素。
操作台與稽核腳本都要用，所以放這裡而不是任一邊的檔案裡。
"""

from __future__ import annotations

import json

OUTLINE = (255, 90, 60)


def parse(boxes_json) -> list[dict]:
    """容錯解析。欄位可能是 NaN（CSV 空值）或還沒定型的鍵名。"""
    if not isinstance(boxes_json, str) or not boxes_json.strip():
        return []
    try:
        raw = json.loads(boxes_json)
    except json.JSONDecodeError:
        return []
    out = []
    for b in raw if isinstance(raw, list) else []:
        box = (b.get("box") or b.get("boundingBox") or []) if isinstance(b, dict) else []
        if len(box) == 4:
            out.append({"box": [float(v) for v in box], "label": b.get("label"), "conf": b.get("conf")})
    return out


def area(boxes_json) -> float:
    """最大框佔畫面的比例。接近 1 表示它框了整張圖，那種框拿來裁圖等於沒裁。"""
    return max(
        (abs(b["box"][2] - b["box"][0]) * abs(b["box"][3] - b["box"][1]) / 1e6 for b in parse(boxes_json)),
        default=0.0,
    )


def draw(path: str, boxes_json: str):
    """把框畫上去。驗證 prompt「框標的物、不要框人」有沒有生效的唯一辦法——
    光看 label 文字看不出它框到哪裡，也看不出框是不是幾乎整張圖。"""
    from PIL import Image, ImageDraw

    im = Image.open(path).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    for b in parse(boxes_json):
        x0, y0, x1, y1 = (
            b["box"][0] * w / 1000,
            b["box"][1] * h / 1000,
            b["box"][2] * w / 1000,
            b["box"][3] * h / 1000,
        )
        d.rectangle([x0, y0, x1, y1], outline=OUTLINE, width=max(2, w // 200))
    return im
