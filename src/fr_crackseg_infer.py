"""crack-seg 預訓練權重 × 樂氧森 469 張：轉移能力推導（零訓練）。

問題：Ultralytics crack-seg（近拍裂縫集）訓練的 yolov8n-seg，能不能轉移到
我們的場域照片（遠景、缺失佔畫面一角）？答案決定 v1 走「公開集預訓練→
微調」還是「自建資料（CVAT）」。

    uv run --extra train src/fr_crackseg_infer.py

輸出：reports/yolo-smoke/crack-seg-on-leyangsen/
    preds/{photoId}.json   每框 {box, conf, mask_area_pct, photo, displayId}
    summary.json           命中率、conf 分布、面積分布、依描述關鍵字分層
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

from paths import FR_MANIFEST, FR_PHOTOS, ROOT, ensure_dirs

ensure_dirs()

WEIGHTS = ROOT / "reports" / "yolo-smoke" / "crack-seg" / "weights" / "best.pt"
OUT = ROOT / "reports" / "yolo-smoke" / "crack-seg-on-leyangsen"
TINY = ROOT / "data" / "field_reports" / "derived" / "tiny_images.json"

CONF = 0.25          # ultralytics 預設 IoU 0.7 / conf 自訂 0.25 起
IMG_BATCH = 8


def main() -> None:
    from ultralytics import YOLO

    tiny = set()
    if TINY.exists():
        tiny = {r["file"] for r in json.loads(TINY.read_text())}

    with open(FR_MANIFEST, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["file"] not in tiny]
    print(f"photos: {len(rows)} (tiny excluded {len(tiny)})")

    model = YOLO(str(WEIGHTS))
    (OUT / "preds").mkdir(parents=True, exist_ok=True)

    import numpy as np  # noqa: F401  (ultralytics 需要)
    from PIL import Image

    all_preds = []
    t0 = time.time()
    files = [FR_PHOTOS / r["file"] for r in rows]
    # ultralytics 一次吃一張路徑最穩（回傳對齊好辦）；批次加速從簡
    for i, (r, path) in enumerate(zip(rows, files, strict=True)):
        res = model.predict(str(path), conf=CONF, imgsz=640, verbose=False, device="mps")
        det = res[0]
        W, H = det.orig_shape[1], det.orig_shape[0]
        rec = {
            "photoId": r["fileId"],
            "file": r["file"],
            "photoDisplayId": r["photoDisplayId"],
            "description": r["description"],
            "width": W,
            "height": H,
            "boxes": [],
        }
        if det.boxes is not None and len(det.boxes):
            masks = det.masks
            for bi in range(len(det.boxes)):
                box = [round(float(v)) for v in det.boxes.xyxy[bi].tolist()]
                conf = round(float(det.boxes.conf[bi]), 3)
                area_pct = 0.0
                if masks is not None and len(masks):
                    m = masks.data[bi]
                    area_pct = round(float(m.float().mean()) * 100, 2)  # % 畫面
                rec["boxes"].append({"box": box, "conf": conf, "mask_area_pct": area_pct,
                                     "cls": int(det.boxes.cls[bi])})
        (OUT / "preds" / f"{r['fileId']}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        all_preds.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)} …")

    elapsed = time.time() - t0
    n_hit = sum(1 for p in all_preds if p["boxes"])
    n_boxes = sum(len(p["boxes"]) for p in all_preds)
    confs = [b["conf"] for p in all_preds for b in p["boxes"]]
    areas = [b["mask_area_pct"] for p in all_preds for b in p["boxes"]]

    # 依描述關鍵字分層（與 fr_gdino 同一套）
    import re
    desc_pat = {"crack": r"裂|龜裂", "gap": r"縫|隙|收邊|收尾", "water": r"滲|漏|積水|水痕",
                "dirt": r"髒|汙|污|垢", "other_defect": r"破|脫落|掉|斷|壞|鏽|刮"}
    layered = {}
    for name, pat in desc_pat.items():
        photos = [p for p in all_preds if p["description"] and re.search(pat, p["description"])]
        hits = sum(1 for p in photos if p["boxes"])
        layered[name] = {"photos": len(photos), "withPred": hits,
                          "rate": round(hits / max(len(photos), 1), 3)}

    summary = {
        "weights": str(WEIGHTS),
        "conf": CONF, "imgsz": 640,
        "photos": len(all_preds), "photosWithPred": n_hit,
        "hitRate": round(n_hit / len(all_preds), 3),
        "boxes": n_boxes,
        "confMean": round(statistics.mean(confs), 3) if confs else 0,
        "confMedian": round(statistics.median(confs), 3) if confs else 0,
        "maskAreaMedianPct": round(statistics.median(areas), 2) if areas else 0,
        "elapsedS": round(elapsed, 1),
        "msPerPhoto": round(elapsed * 1000 / len(all_preds)),
        "layeredByDescription": layered,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
