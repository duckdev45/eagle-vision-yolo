"""階段 B：四角遮蔽 + 統一格式。

SPEC §7.2 的鐵律：前處理必須與標籤完全無關。
遮蔽依「這張圖上有沒有烤東西」決定，不依標籤：
  - reportDate < CLEAN_FROM：上傳端把日報膠囊烤在畫面上 → 遮四角。
  - reportDate >= CLEAN_FROM：上傳端改存乾淨 raw → 不遮，遮了只是砍掉真畫面。
代價：灰角變成「舊資料」的訊號。日期本來就與工地/工序相關，
所以 split 必須照 constrId+日期切（已是 split_dates 的行為），
不要讓同一天橫跨 train/test，否則模型可以背日期分群。

    uv run src/prepare.py [--force]
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys

from PIL import Image, ImageDraw, ImageFile

from photo_quality import file_problem

# 上傳端偶有截斷的檔（實測 1/525）。丟掉整張不划算，殘缺的下緣照樣能訓練。
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.dirname(__file__))
import paths  # noqa: E402

CORNER_W, CORNER_H = 0.30, 0.12  # 日報膠囊：四角
LONG_EDGE = 512
FILL = (127, 127, 127)
CLEAN_FROM = "2026-08-14"  # 這天起日報照存乾淨 raw，畫面上不再有膠囊


def clean_ids() -> set[str]:
    """reportDate 落在 [CLEAN_FROM, syncedAt] 的日報照＝乾淨 raw，無膠囊可遮。

    上界用該列的 syncedAt：日報日期不可能晚於同步時間，這樣髒的未來日期
    （實測有 2085-04-20）不會被誤判成乾淨圖而漏遮。
    """
    if not paths.MANIFEST.exists():
        return set()
    with open(paths.MANIFEST, newline="", encoding="utf-8") as f:
        return {r["fileId"] for r in csv.DictReader(f) if is_clean_report(r["reportDate"], r["syncedAt"])}


def is_clean_report(report_date: str, synced_at: str) -> bool:
    """新日報原圖無膠囊；未來日期的髒資料仍按舊圖遮蔽。"""
    return CLEAN_FROM <= report_date <= synced_at[:10]


def mask_corners(
    im: Image.Image, w_frac: float = CORNER_W, h_frac: float = CORNER_H, fill=FILL
) -> Image.Image:
    """日報：四角膠囊 + 右下日期標籤。"""
    w, h = im.size
    cw, ch = int(w * w_frac), int(h * h_frac)
    d = ImageDraw.Draw(im)
    for x0, y0 in ((0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch)):
        d.rectangle([x0, y0, x0 + cw, y0 + ch], fill=fill)
    return im


def mask_none(im: Image.Image) -> Image.Image:
    """舊 pptx：照片是原檔貼進投影片的，畫面上沒有系統烤的東西，沒得遮。"""
    return im


# 資料源 → 遮罩。兩邊遮的東西不同，因為兩邊烤上去的東西不同。
# legacy（舊 pptx 照片）只供 G1 黃金集與 G2 考卷的特徵，不進訓練。
# QMS 浮水印遮罩隨 QMS 實驗線於 2026-10-09 移除（資料見 data/archive/README.md）。
MASKS = {"report": mask_corners, "legacy": mask_none}
SRC_DIRS = {
    "report": (paths.PHOTOS, paths.IMAGES),
    "legacy": (paths.LEGACY_PHOTOS, paths.LEGACY_IMAGES),
}


def prepare_jpeg(src, kind: str = "report", mask: bool = True) -> bytes:
    """批次前處理與即時推論共用的位元組結果。"""
    with Image.open(src) as original:
        im = original.convert("RGB")
    if max(im.size) > LONG_EDGE:
        im.thumbnail((LONG_EDGE, LONG_EDGE), Image.LANCZOS)
    if mask:
        MASKS[kind](im)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=90)
    return out.getvalue()


def process(src, dst, kind: str = "report", mask: bool = True) -> None:
    dst.write_bytes(prepare_jpeg(src, kind, mask))


def run(force: bool = False, kind: str = "report", log=print) -> dict:
    paths.ensure_dirs()
    src_dir, out_dir = SRC_DIRS[kind]
    clean = clean_ids() if kind == "report" else set()
    done = skipped = failed = unmasked = invalid = 0
    for src in sorted(src_dir.iterdir()):
        if not src.is_file() or src.name.startswith("."):
            continue
        dst = out_dir / f"{src.stem}.jpg"
        if dst.exists() and not force:
            skipped += 1
            continue
        problem = file_problem(src) if kind == "report" else None
        if problem:
            log(f"  跳過壞檔 {src.name}: {problem}（原檔保留，等待同步補抓）")
            invalid += 1
            continue
        try:
            is_clean = src.stem in clean
            process(src, dst, kind, mask=not is_clean)
            done += 1
            unmasked += is_clean
        except Exception as e:
            log(f"  失敗 {src.name}: {e}")
            failed += 1
    stat = {"processed": done, "skipped": skipped, "failed": failed, "unmasked": unmasked, "invalid": invalid}
    log(f"前處理完成：{stat}")
    return stat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--src", default="report", choices=["report", "legacy", "all"])
    a = ap.parse_args()
    for k in list(SRC_DIRS) if a.src == "all" else [a.src]:
        run(a.force, k)
