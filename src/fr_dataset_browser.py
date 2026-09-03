"""資料集 GT 瀏覽器 manifest 生成：index.html 消費。

每集輸出「畫好 GT 的縮圖」+ 原圖（放大用）路徑：
- crack-seg：分割多邊形（紅）
- MBDD2025：bbox 5 類（各類配色＋label）
- tile-damage：無標註

縮圖上直接畫 GT（先縮再畫，線寬才看得見——教訓見 fr_dataset_preview.py）；
原檔以 file:// 供 lightbox 放大（瀏覽器直接讀，零損失）。

    uv run src/fr_dataset_browser.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path("/Users/duck/PycharmProjects/eagle-vision/reports/yolo-smoke/dataset-preview")
THUMBS = OUT / "thumbs"
THUMBS.mkdir(parents=True, exist_ok=True)

MBDD_NAMES = {0: "crack", 1: "leakage", 2: "abscission", 3: "corrosion", 4: "bulge"}
MBDD_COLORS = {0: (225, 29, 72), 1: (14, 165, 233), 2: (249, 115, 22),
               3: (124, 58, 237), 4: (5, 150, 105)}
N_PER_SET = 60          # 每集先鋪 60 張（載入快；要全量再調）
THUMB_SIDE = 640


def poly_points(pts, W, H):
    return [(float(pts[i]) * W, float(pts[i + 1]) * H) for i in range(0, len(pts), 2)]


def make_thumb(src: Path, draw_fn, name: str) -> str:
    """縮圖＋畫 GT；回傳縮圖路徑。"""
    dst = THUMBS / f"{name}.jpg"
    if dst.exists():
        return f"thumbs/{name}.jpg"
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_SIDE, THUMB_SIDE))
        W, H = im.size
        dr = ImageDraw.Draw(im, "RGBA")
        draw_fn(dr, W, H)
        im.save(dst, quality=82)
    return f"thumbs/{name}.jpg"


def crackseg_items() -> list[dict]:
    root = Path("/Users/duck/datasets/crack-seg")
    files = sorted((root / "images" / "train").glob("*"))
    files = [f for f in files if (root / "labels" / "train" / (f.stem + ".txt")).exists()]
    rng = random.Random(7)
    picks = rng.sample(files, min(N_PER_SET, len(files)))
    out = []
    for f in picks:
        lbl = root / "labels" / "train" / (f.stem + ".txt")
        def draw(dr, W, H, lbl=lbl):
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) >= 8:
                    dr.polygon(poly_points(p[1:], W, H),
                               fill=(225, 29, 72, 55), outline=(225, 29, 72, 255), width=4)
        out.append({
            "f": make_thumb(f, draw, f"cs_{f.stem[:40]}"),
            "orig": str(f),
            "cap": f"crack-seg · {f.stem[:24]}",
        })
    return out


def mbdd_items() -> list[dict]:
    root = Path("/Users/duck/datasets/MBDD2025/MBDD2025")
    ids = [l.strip() for l in (root / "train.txt").read_text().splitlines() if l.strip()]
    rng = random.Random(11)
    labeled = []
    for rel in ids:
        lbl_p = root / "Labels" / (Path(rel).stem + ".txt")
        if lbl_p.exists() and any(len(l.split()) == 5 for l in lbl_p.read_text().splitlines()):
            labeled.append(rel)
        if len(labeled) >= 400:
            break
    picks = rng.sample(labeled, min(N_PER_SET, len(labeled)))
    out = []
    for rel in picks:
        img_p = root / "JPEGImages" / Path(rel).name
        lbl_p = root / "Labels" / (Path(rel).stem + ".txt")
        def draw(dr, W, H, lbl_p=lbl_p):
            for line in lbl_p.read_text().splitlines():
                p = line.split()
                if len(p) == 5:
                    cls, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                    x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
                    x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
                    c = MBDD_COLORS.get(cls, (100, 100, 100))
                    dr.rectangle([x1, y1, x2, y2], outline=c, width=5)
                    dr.text((x1 + 7, y1 + 7), MBDD_NAMES.get(cls, str(cls)), fill=c)
        stem = Path(rel).stem
        out.append({
            "f": make_thumb(img_p, draw, f"mb_{stem[:40]}"),
            "orig": str(img_p),
            "cap": f"MBDD2025 · {stem}",
        })
    return out


def tile_items() -> list[dict]:
    root = Path("/Users/duck/datasets/tile-damage/damage_detection.v85i.yolov7/train/images")
    files = sorted(root.glob("*.jpg"))
    rng = random.Random(3)
    picks = rng.sample(files, min(N_PER_SET, len(files)))
    out = []
    for f in picks:
        out.append({
            "f": make_thumb(f, lambda dr, W, H: None, f"td_{f.stem[:40]}"),
            "orig": str(f),
            "cap": f"tile-damage · {f.stem[:22]}（無 GT）",
        })
    return out


def main() -> None:
    manifest = {
        "crackseg": crackseg_items(),
        "mbdd": mbdd_items(),
        "tile": tile_items(),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in manifest.items()}, "->", OUT / "manifest.json")


if __name__ == "__main__":
    main()
