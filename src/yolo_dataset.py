"""YOLO11 資料集建置器（ROADMAP 執行序 #8 的程式前置——框一到位，當天開訓）。

框源鐵律（yolo_preprocessor 哲學寫成硬條款）：**訓練集僅收 HUMAN 層框**。
  - 正式來源：data/defects.csv 裡 source=cvat* 的框（CVAT 匯入，`src/cvat_import.py`）
  - 量測來源：GDINO 人審 verdict=correct 的框（--measure-gdino）——只餵 imgsz
    幾何量測，不產生訓練集（16 框過不了 min_boxes，樣態也是英文詞彙表）

split 鐵律：**案場×日期整組切**（split.py 同一條規則）。同案場同日的缺失有視覺
相關性（同工班、同批材料、同光源），按單張照片切，val 的 mAP 會虛高。單一群組
（如樂氧森單日）無法組切時會明寫警告並退化成隨機切——僅供量測，不可出正式數字。

imgsz/tiling 量測（前置決策要求、一直沒人做）：每個框換算成 imgsz 640/1280 下的
**像素**尺寸，統計 <8px/<16px（YOLO 小物件線）比例——比例高就該調 imgsz 或上 tiling，
別讓細裂縫在縮圖裡消失。

    uv run src/yolo_dataset.py --out v1-defects          # defects.csv → 資料集＋量測
    uv run src/yolo_dataset.py --measure-gdino           # 只跑量測（不建資料集）

產出：data/yolo/{out}/  images/ labels/ data.yaml + stats.json（量測＋切分統計）
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from core import paths
from core.defects import load_defects


def collect_cvat_boxes() -> pd.DataFrame:
    """defects.csv 的 CVAT 框 → [fileId, pattern, x0..y1(0~1000), source]。"""
    df = load_defects()
    df = df[df.source.str.startswith("cvat") & df.box.astype(bool)]
    rows = []
    for _, r in df.iterrows():
        try:
            for one in json.loads(r.box):
                if len(one) == 4:
                    rows.append(
                        {
                            "fileId": r.fileId,
                            "pattern": r.defectType,
                            "x0": one[0],
                            "y0": one[1],
                            "x1": one[2],
                            "y1": one[3],
                            "source": r.source,
                        }
                    )
        except Exception:
            continue
    return pd.DataFrame(rows)


def collect_gdino_correct() -> pd.DataFrame:
    """GDINO 人審 verdict=correct 的框（像素座標 → 0~1000），帶原圖尺寸。量測用，不可訓練。"""
    v = pd.read_csv(paths.FIELD_REPORTS / "derived" / "gdino" / "verdicts.csv")
    v = v[v.verdict == "correct"]
    rows = []
    for run_id, g in v.groupby("runId"):
        anno_dir = paths.FR_GDINO / "runs" / str(run_id) / "annotations"
        for _, r in g.iterrows():
            p = anno_dir / f"{r.photoId}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            boxes = d.get("boxes") or []
            if r.boxIdx >= len(boxes):
                continue
            b = boxes[r.boxIdx]
            x0, y0, x1, y1 = b["box"]
            w, h = d["width"], d["height"]
            rows.append(
                {
                    "fileId": r.photoId,
                    "pattern": b["pattern"],
                    "x0": x0 / w * 1000,
                    "y0": y0 / h * 1000,
                    "x1": x1 / w * 1000,
                    "y1": y1 / h * 1000,
                    "w": w,
                    "h": h,
                    "source": f"gdino-correct/{run_id}",
                }
            )
    return pd.DataFrame(rows)


def photo_file(file_id: str) -> Path | None:
    for p in (
        paths.FR_PHOTOS / f"{file_id}.webp",
        *(paths.PHOTOS / f"{file_id}{e}" for e in (".webp", ".jpg", ".png")),
        *(paths.LEGACY_PHOTOS / f"{file_id}{e}" for e in (".jpg", ".png", ".webp")),
    ):
        if p.exists():
            return p
    return None


def group_of(file_id: str) -> str:
    """案場×日期。樂氧森在 field_reports manifest、日報在 manifest。"""
    fr = pd.read_csv(
        paths.FIELD_REPORTS / "raw" / "manifest.csv", dtype=str, keep_default_na=False, na_values=[""]
    )
    hit = fr[fr.fileId == file_id]
    if len(hit):
        return "樂氧森|" + str(hit.reportDate.iloc[0])
    daily = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
    hit = daily[daily.fileId == file_id]
    if len(hit):
        return str(hit.constrId.iloc[0]) + "|" + str(hit.reportDate.iloc[0])
    return "unknown"


def imgsz_report(df: pd.DataFrame, dims: dict[str, tuple[int, int]] | None = None) -> dict:
    """每框在 640/1280 下的像素尺寸；<8/<16px 的比例——小物件會不會被縮沒。

    dims 缺的部分從照片檔現讀（PIL）。
    """
    from PIL import Image

    dims = dict(dims or {})
    need = [f for f in df.fileId.unique() if not dims.get(f)]
    for f in need:
        p = photo_file(f)
        if p is None:
            continue
        try:
            dims[f] = Image.open(p).size
        except Exception:
            continue
    out = {}
    for imgsz in (640, 1280):
        px = []
        for _, r in df.iterrows():
            w, h = dims.get(r.fileId, (0, 0))
            if not w or not h:
                continue
            scale = imgsz / max(w, h)
            px.append(min((r.x1 - r.x0) / 1000 * w * scale, (r.y1 - r.y0) / 1000 * h * scale))
        if not px:
            continue
        s = pd.Series(px)
        out[f"imgsz{imgsz}"] = {
            "boxes": len(px),
            "median_px": round(float(s.median()), 1),
            "p10_px": round(float(s.quantile(0.1)), 1),
            "lt8px": round(float((s < 8).mean()), 3),
            "lt16px": round(float((s < 16).mean()), 3),
        }
    out["verdict"] = (
        "imgsz640 的 <16px 比例 <5% → 640 可用"
        if out.get("imgsz640", {}).get("lt16px", 1) < 0.05
        else "imgsz640 有大量小物件被縮沒——調 1280 或上 tiling 再訓"
    )
    return out


def build(
    boxes: pd.DataFrame, out_name: str, min_boxes: int, val_frac: float, seed: int, smoke_note: str = ""
) -> dict | None:
    if boxes.empty:
        return None
    counts = boxes.pattern.value_counts()
    keep = counts[counts >= min_boxes].index.tolist()
    dropped = counts[counts < min_boxes]
    if dropped.any():
        print(f"  類別不足 {min_boxes} 框，整批不進 v1：{dict(dropped)}（ROADMAP 前置決策）")
    boxes = boxes[boxes.pattern.isin(keep)]
    if boxes.empty:
        print("  沒有任何樣態過門檻——不建資料集。")
        return None

    groups = boxes.fileId.map(group_of)
    boxes = boxes.assign(group=groups)
    uniq = boxes.group.unique()
    rng = random.Random(seed)
    shuffled = sorted(uniq)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    val_groups = set(shuffled[:n_val])
    degenerate = len(uniq) < 2
    if degenerate:
        print(f"  ⚠ 只有 {len(uniq)} 個案場×日期群組——無法組切，退化成隨機切（僅供量測，不可出正式數字）")
        ids = boxes.fileId.unique().tolist()
        rng.shuffle(ids)
        val_ids = set(ids[: round(len(ids) * val_frac)])
        boxes = boxes.assign(isVal=boxes.fileId.isin(val_ids))
    else:
        print(f"  切分：{len(uniq)} 群組（案場×日期），val {n_val} 組")
        boxes = boxes.assign(isVal=boxes.group.isin(val_groups))

    out = paths.DERIVED / "yolo" / out_name
    if out.exists():
        raise SystemExit(f"{out} 已存在——換名字，別覆寫已產出的資料集")
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    names = {c: i for i, c in enumerate(sorted(keep))}
    dims: dict[str, tuple[int, int]] = {}
    n_img = Counter()
    for (fid, is_val), g in boxes.groupby(["fileId", "isVal"]):
        src = photo_file(fid)
        if src is None:
            print(f"  跳過 {fid[:8]}：找不到照片檔")
            continue
        split = "val" if is_val else "train"
        dst = out / "images" / split / f"{fid}.jpg"
        if not dst.exists():
            shutil.copy(src, dst)
        from PIL import Image

        w, h = Image.open(dst).size
        dims[fid] = (w, h)
        with (out / "labels" / split / f"{fid}.txt").open("w") as f:
            for _, r in g.iterrows():
                cx, cy = (r.x0 + r.x1) / 2000, (r.y0 + r.y1) / 2000
                bw, bh = (r.x1 - r.x0) / 1000, (r.y1 - r.y0) / 1000
                f.write(f"{names[r.pattern]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        n_img[split] += 1

    yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "names": {v: k for k, v in names.items()},
    }
    (out / "data.yaml").write_text(json.dumps(yaml, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "smoke_note": smoke_note,
        "boxes_total": len(boxes),
        "images": dict(n_img),
        "classes": {k: int(counts[k]) for k in sorted(keep)},
        "dropped_classes": {k: int(v) for k, v in dropped.items()},
        "groups": {"total": len(uniq), "val": n_val, "degenerate_split": degenerate},
        "imgsz": imgsz_report(boxes, dims),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="v1-defects")
    ap.add_argument(
        "--min-boxes", type=int, default=30, help="樣態框數下限，不足整批不進（ROADMAP 前置決策）"
    )
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--measure-gdino", action="store_true", help="只跑 GDINO 人審框的 imgsz 量測，不建資料集")
    a = ap.parse_args()

    cvat = collect_cvat_boxes()
    print(f"defects.csv 的 CVAT 框：{len(cvat)}")
    if a.measure_gdino:
        g = collect_gdino_correct()
        print(f"GDINO 人審 correct 框（量測用）：{len(g)}")
        if not g.empty:
            print(json.dumps(imgsz_report(g), ensure_ascii=False, indent=1))
        return 0
    if cvat.empty:
        print("defects.csv 還沒有 CVAT 框——CVAT 冷啟動標完、`src/cvat_import.py` 匯入後再來。")
        print("想先看 imgsz 量測的話：--measure-gdino")
        return 0
    build(cvat, a.out, a.min_boxes, a.val_frac, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
