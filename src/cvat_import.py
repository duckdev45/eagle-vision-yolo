"""CVAT 匯入：CVAT for images 1.1 XML → data/defects.csv（HUMAN 層框）。

匯出端（src/cvat_export.py）把照片命名成 fileId.jpg，所以 XML 的 image name
去掉附檔名就是 fileId——對應不會斷。座標是像素，按 image 的 width/height
換算成 0~1000 相對座標（與 review.csv 的 box 同格式）。

    uv run src/cvat_import.py --xml "data/cvat/coldstart-*/export/*.xml" --annotator 名字

- label 必須是 core.defects.DEFECT_PATTERNS 裡的 10 樣態之一，其他（含「其他」）
  一律拒收並列出來——拼錯的 label 靜默入庫，YOLO 訓練就會多出一個幽靈類。
- 去重鍵 = (fileId, source, defectType, box)，同一份 XML 匯兩次不會翻倍。
"""

from __future__ import annotations

import argparse
import glob
import sys as _sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))

from core.defects import DEFECT_PATTERNS, append_rows


def parse_cvat_xml(xml_path: str) -> list[dict]:
    """一個 XML → 原始框列（fileId, defectType, box 0~1000）。座標夾到邊界內。"""
    root = ET.parse(xml_path).getroot()
    rows: list[dict] = []
    for img in root.iter("image"):
        name = img.get("name", "")
        file_id = Path(name).stem
        try:
            w, h = float(img.get("width", 0)), float(img.get("height", 0))
        except (TypeError, ValueError):
            w = h = 0.0
        if w <= 0 or h <= 0:
            print(f"  跳過 {name}：缺 width/height")
            continue
        for box in img.iter("box"):
            label = (box.get("label") or "").strip()
            try:
                xtl, ytl = float(box.get("xtl", 0)), float(box.get("ytl", 0))
                xbr, ybr = float(box.get("xbr", 0)), float(box.get("ybr", 0))
            except (KeyError, TypeError, ValueError):
                print(f"  跳過 {name} 一框：座標不完整")
                continue
            # CVAT 框可能在圖外（畫出界），夾回 0~1000
            x0 = max(0, min(1000, round(xtl / w * 1000)))
            y0 = max(0, min(1000, round(ytl / h * 1000)))
            x1 = max(0, min(1000, round(xbr / w * 1000)))
            y1 = max(0, min(1000, round(ybr / h * 1000)))
            if x1 - x0 < 6 or y1 - y0 < 6:  # 誤點小框（同 review_ui 的 6/1000 門檻）
                continue
            rows.append({"fileId": file_id, "defectType": label, "box": [x0, y0, x1, y1]})
    return rows


def import_xml(xml_paths: list[str], annotator: str, source: str = "cvat") -> int:
    valid = set(DEFECT_PATTERNS)
    rows: list[dict] = []
    rejected: dict[str, int] = {}
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for xp in xml_paths:
        for r in parse_cvat_xml(xp):
            if r["defectType"] not in valid:
                rejected[r["defectType"]] = rejected.get(r["defectType"], 0) + 1
                continue
            rows.append(
                {
                    "fileId": r["fileId"],
                    "source": source,
                    "defectType": r["defectType"],
                    "note": Path(xp).name,
                    "reviewedBy": annotator,
                    "reviewedAt": now,
                    "box": __import__("json").dumps([r["box"]], ensure_ascii=False),
                }
            )
    if rejected:
        print(f"  拒收非樣態 label：{rejected}（不在 10 樣態清單——先在 CVAT 改 label 再匯）")
    if not rows:
        print("沒有可匯入的框。")
        return 0
    n = append_rows(rows)
    by = {}
    for r in rows:
        by[r["defectType"]] = by.get(r["defectType"], 0) + 1
    print(f"匯入 {n} 框（重複 {len(rows) - n}）· 各樣態：{dict(sorted(by.items(), key=lambda x: -x[1]))}")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True, help="CVAT 1.1 XML（可吃 glob）")
    ap.add_argument("--annotator", required=True, help="標註者名字（進 reviewedBy）")
    ap.add_argument("--source", default="cvat", help="defects.csv 的 source 欄")
    a = ap.parse_args()
    paths_ = sorted(glob.glob(a.xml))
    if not paths_:
        raise SystemExit(f"glob 沒命中：{a.xml}")
    print(f"{len(paths_)} 個 XML")
    import_xml(paths_, a.annotator, a.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
