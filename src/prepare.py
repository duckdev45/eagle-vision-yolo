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
import os
import sys

from PIL import Image, ImageDraw, ImageFile

# 上傳端偶有截斷的檔（實測 1/525）。丟掉整張不划算，殘缺的下緣照樣能訓練。
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.dirname(__file__))
import paths  # noqa: E402

CORNER_W, CORNER_H = 0.30, 0.12   # 日報膠囊：四角
# QMS 浮水印：實測 x 2%~52% / y 65%~97%，烤死在原檔上，沒有乾淨版本。
# 留 margin 吸收機型解析度差異。
WATERMARK = (0.0, 0.60, 0.56, 1.0)   # (x0, y0, x1, y1) 比例
LONG_EDGE = 512
FILL = (127, 127, 127)
CLEAN_FROM = "2026-08-14"   # 這天起日報照存乾淨 raw，畫面上不再有膠囊


def clean_ids() -> set[str]:
    """reportDate 落在 [CLEAN_FROM, syncedAt] 的日報照＝乾淨 raw，無膠囊可遮。

    上界用該列的 syncedAt：日報日期不可能晚於同步時間，這樣髒的未來日期
    （實測有 2085-04-20）不會被誤判成乾淨圖而漏遮。
    """
    if not paths.MANIFEST.exists():
        return set()
    with open(paths.MANIFEST, newline="", encoding="utf-8") as f:
        return {r["fileId"] for r in csv.DictReader(f)
                if CLEAN_FROM <= r["reportDate"] <= r["syncedAt"][:10]}


def mask_corners(im: Image.Image, w_frac: float = CORNER_W, h_frac: float = CORNER_H,
                 fill=FILL) -> Image.Image:
    """日報：四角膠囊 + 右下日期標籤。"""
    w, h = im.size
    cw, ch = int(w * w_frac), int(h * h_frac)
    d = ImageDraw.Draw(im)
    for x0, y0 in ((0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch)):
        d.rectangle([x0, y0, x0 + cw, y0 + ch], fill=fill)
    return im


def mask_watermark(im: Image.Image, fill=FILL) -> Image.Image:
    """QMS：只有左下浮水印（實測 x 2%~52% / y 65%~97%，留 margin）。"""
    w, h = im.size
    x0, y0, x1, y1 = WATERMARK
    ImageDraw.Draw(im).rectangle([int(w * x0), int(h * y0), int(w * x1), int(h * y1)], fill=fill)
    return im


def mask_none(im: Image.Image) -> Image.Image:
    """舊 pptx：照片是原檔貼進投影片的，畫面上沒有系統烤的東西，沒得遮。"""
    return im


# 資料源 → 遮罩。三邊遮的東西不同，因為三邊烤上去的東西不同。
# 這在**兩階段微調**下是安全的：QMS 預訓練與 PMS 微調的資料不在同一個分類器裡競爭，
# 遮罩形狀無從變成「哪個系統 → 哪個工種」的捷徑。若哪天要單一分類器混訓，這條要重談。
MASKS = {"report": mask_corners, "qms": mask_watermark, "legacy": mask_none}
SRC_DIRS = {
    "report": (paths.PHOTOS, paths.IMAGES),
    "qms": (paths.QMS_PHOTOS, paths.QMS_IMAGES),
    "legacy": (paths.LEGACY_PHOTOS, paths.LEGACY_IMAGES),
}


def process(src, dst, kind: str = "report", mask: bool = True) -> None:
    im = Image.open(src).convert("RGB")
    if max(im.size) > LONG_EDGE:
        im.thumbnail((LONG_EDGE, LONG_EDGE), Image.LANCZOS)
    if mask:
        MASKS[kind](im)
    im.save(dst, "JPEG", quality=90)


def run(force: bool = False, kind: str = "report", log=print) -> dict:
    paths.ensure_dirs()
    src_dir, out_dir = SRC_DIRS[kind]
    clean = clean_ids() if kind == "report" else set()   # QMS 浮水印照樣烤死，全遮
    done = skipped = failed = unmasked = 0
    for src in sorted(src_dir.iterdir()):
        if not src.is_file() or src.name.startswith("."):
            continue
        dst = out_dir / f"{src.stem}.jpg"
        if dst.exists() and not force:
            skipped += 1
            continue
        try:
            is_clean = src.stem in clean
            process(src, dst, kind, mask=not is_clean)
            done += 1
            unmasked += is_clean
        except Exception as e:
            log(f"  失敗 {src.name}: {e}")
            failed += 1
    stat = {"processed": done, "skipped": skipped, "failed": failed, "unmasked": unmasked}
    log(f"前處理完成：{stat}")
    return stat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--src", default="report",
                    choices=["report", "qms", "legacy", "all"])
    a = ap.parse_args()
    for k in (list(SRC_DIRS) if a.src == "all" else [a.src]):
        run(a.force, k)
