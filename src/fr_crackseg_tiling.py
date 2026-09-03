"""imgsz/tiling 政策煙霧測試（ROADMAP ①「imgsz/tiling 政策在煙霧測試一併量測」）。

raw 原圖長邊幾千畫素，YOLO imgsz 預設 640 會把細裂縫縮沒。三種設定量同一批
樂氧森真實照片，比命中率／框數／mask 面積分布，才知道要不要付 tiling 的複雜度：

    baseline   imgsz=640（現況，fr_crackseg_infer.py 同設定）
    bigger     imgsz=1280（不切圖，只放大輸入）
    tiled      切 800px 重疊 20% 小塊分別推論，框座標還原＋NMS 合併跨塊重複框

    uv run --extra train src/fr_crackseg_tiling.py

輸出：reports/yolo-smoke/crack-seg-tiling/summary.json
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
OUT = ROOT / "reports" / "yolo-smoke" / "crack-seg-tiling"
TINY = ROOT / "data" / "field_reports" / "derived" / "tiny_images.json"

CONF = 0.25
TILE = 800       # 拍板值：配合 640 輸入模型留邊界重疊空間
OVERLAP = 0.2    # 拍板值：避免裂縫切在邊界兩邊都抓不完整
NMS_IOU = 0.5


def tile_boxes(W: int, H: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """回傳 (x0,y0,x1,y1) 切塊清單，短邊小於 tile 時整張只算一塊。"""
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y, min(x + tile, W), min(y + tile, H)) for y in ys for x in xs]


def nms_merge(boxes, confs, areas, iou_thr):
    import torch
    from torchvision.ops import nms

    if not boxes:
        return [], [], []
    b = torch.tensor(boxes, dtype=torch.float32)
    c = torch.tensor(confs, dtype=torch.float32)
    keep = nms(b, c, iou_thr).tolist()
    return [boxes[i] for i in keep], [confs[i] for i in keep], [areas[i] for i in keep]


def predict_tiled(model, path: Path, tile: int, overlap: float, conf: float):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    W, H = im.size
    boxes, confs, areas = [], [], []
    for x0, y0, x1, y1 in tile_boxes(W, H, tile, overlap):
        crop = im.crop((x0, y0, x1, y1))
        res = model.predict(crop, conf=conf, imgsz=tile, verbose=False, device="mps")[0]
        if res.boxes is None or not len(res.boxes):
            continue
        masks = res.masks
        tw, th = x1 - x0, y1 - y0
        for bi in range(len(res.boxes)):
            bx = res.boxes.xyxy[bi].tolist()
            boxes.append([bx[0] + x0, bx[1] + y0, bx[2] + x0, bx[3] + y0])
            confs.append(float(res.boxes.conf[bi]))
            area_pct = 0.0
            if masks is not None and len(masks):
                area_pct = float(masks.data[bi].float().mean()) * (tw * th) / (W * H) * 100
            areas.append(area_pct)
    boxes, confs, areas = nms_merge(boxes, confs, areas, NMS_IOU)
    return boxes, confs, areas


def predict_plain(model, path: Path, imgsz: int, conf: float):
    res = model.predict(str(path), conf=conf, imgsz=imgsz, verbose=False, device="mps")[0]
    boxes, confs, areas = [], [], []
    if res.boxes is not None and len(res.boxes):
        masks = res.masks
        for bi in range(len(res.boxes)):
            boxes.append(res.boxes.xyxy[bi].tolist())
            confs.append(float(res.boxes.conf[bi]))
            areas.append(float(masks.data[bi].float().mean()) * 100 if masks is not None and len(masks) else 0.0)
    return boxes, confs, areas


def summarize(name: str, per_photo: list[dict]) -> dict:
    n_hit = sum(1 for p in per_photo if p["boxes"])
    n_boxes = sum(p["boxes"] for p in per_photo)
    confs = [c for p in per_photo for c in p["confs"]]
    areas = [a for p in per_photo for a in p["areas"] if a > 0]
    return {
        "setting": name,
        "photos": len(per_photo),
        "photosWithPred": n_hit,
        "hitRate": round(n_hit / len(per_photo), 3),
        "boxes": n_boxes,
        "confMean": round(statistics.mean(confs), 3) if confs else 0,
        "maskAreaMedianPct": round(statistics.median(areas), 3) if areas else 0,
    }


def main() -> None:
    from ultralytics import YOLO

    tiny = set()
    if TINY.exists():
        tiny = {r["file"] for r in json.loads(TINY.read_text())}
    with open(FR_MANIFEST, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["file"] not in tiny]
    print(f"photos: {len(rows)} (tiny excluded {len(tiny)})")

    model = YOLO(str(WEIGHTS))
    OUT.mkdir(parents=True, exist_ok=True)

    settings = {
        "imgsz640": lambda p: predict_plain(model, p, 640, CONF),
        "imgsz1280": lambda p: predict_plain(model, p, 1280, CONF),
        f"tiled{TILE}_{int(OVERLAP*100)}pct": lambda p: predict_tiled(model, p, TILE, OVERLAP, CONF),
    }

    results = {}
    for name, fn in settings.items():
        t0 = time.time()
        per_photo = []
        for i, r in enumerate(rows):
            boxes, confs, areas = fn(FR_PHOTOS / r["file"])
            per_photo.append({"fileId": r["fileId"], "boxes": len(boxes), "confs": confs, "areas": areas})
            if (i + 1) % 100 == 0:
                print(f"  [{name}] {i+1}/{len(rows)} …")
        s = summarize(name, per_photo)
        s["elapsedS"] = round(time.time() - t0, 1)
        s["msPerPhoto"] = round((time.time() - t0) * 1000 / len(rows))
        results[name] = s
        print(json.dumps(s, ensure_ascii=False, indent=1))

    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
