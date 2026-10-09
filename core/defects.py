"""缺失軸的資料存取（data/defects.csv）。

契約在 docs/specs/reject-qs-code.md §2：退回（app）／LINE／CVAT 三軌寫同一張表，
`source` 分流；與 review.csv（工種 cls 裁決）**分開**——缺失與工種是正交維度。

- 照片層級的缺陷回報（退回/LINE）：defectType 必填、box 空。
- 框層級的標註（CVAT 匯出）：一行一框，defectType = 缺失樣態、box = [[x0,y0,x1,y1]]
  （0~1000 相對原圖，與 review.csv 的 box 同格式，`boxes.area`/`explain.iou` 直接可用）。
- append-only；去重鍵 = (fileId, source, defectType, box字串)——同照片同框重複匯入不會翻倍。
- status 一律 HUMAN 層（人選/人框的），AI_GUESS 不寫這張表（yolo_preprocessor 哲學）。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from core import paths

COLUMNS = [
    "fileId",
    "source",
    "site",
    "location",
    "trade",
    "defectType",
    "qsCode",
    "note",
    "reviewedBy",
    "reviewedAt",
    "box",
]

# 缺失樣態：與退回按鈕 spec 的 defectType enum 一致（工種正交，順序＝按鈕選單順序）。
# 「其他」「不確定」是按鈕專用（單選必填的逃生口），CVAT 標框不收——框不出來的
# 東西本來就不該有框。
DEFECT_PATTERNS = [
    "縫隙/收邊",
    "髒污/殘留",
    "破損/脫落",
    "滲水/水痕",
    "不平整",
    "保護不足",
    "刮傷/撞痕",
    "鏽蝕",
    "裂縫",
    "掉漆/漆面",
]
BUTTON_ONLY = {"其他", "不確定"}


def load_defects() -> pd.DataFrame:
    if not paths.DEFECTS.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(paths.DEFECTS, dtype=str, keep_default_na=False, na_values=[""])
    for c in COLUMNS:  # 舊檔缺欄位時補齊
        if c not in df.columns:
            df[c] = ""
    return df[COLUMNS]


def _dedup_key(row: dict) -> str:
    return json.dumps(
        [row.get("fileId", ""), row.get("source", ""), row.get("defectType", ""), row.get("box", "")],
        ensure_ascii=False,
    )


def append_rows(rows: list[dict], log=print) -> int:
    """append 一批，回傳實際新增筆數。去重鍵 = (fileId, source, defectType, box)。"""
    for r in rows:
        for c in COLUMNS:
            r.setdefault(c, "")
        r["reviewedAt"] = r.get("reviewedAt") or datetime.now(UTC).isoformat(timespec="seconds")
    seen = {_dedup_key(r) for _, r in load_defects().iterrows()}
    fresh, fresh_keys = [], set()
    for r in rows:
        k = _dedup_key(r)
        if k not in seen and k not in fresh_keys:
            fresh.append({c: r.get(c, "") for c in COLUMNS})
            fresh_keys.add(k)
    if not fresh:
        return 0
    new = pd.DataFrame(fresh, columns=COLUMNS)
    if paths.DEFECTS.exists():
        old = pd.read_csv(paths.DEFECTS, dtype=str, keep_default_na=False, na_values=[""])
        if list(old.columns) != COLUMNS:  # 欄位錯位時整份重寫
            pd.concat([old.reindex(columns=COLUMNS), new], ignore_index=True).to_csv(
                paths.DEFECTS, index=False
            )
        else:
            new.to_csv(paths.DEFECTS, mode="a", header=False, index=False)
    else:
        paths.DEFECTS.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(paths.DEFECTS, index=False)
    log(f"defects.csv +{len(fresh)} 筆")
    return len(fresh)


def boxes_by_pattern() -> dict[str, list[tuple[str, list[int]]]]:
    """{缺失樣態: [(fileId, [x0,y0,x1,y1])]}——只收有框的列，YOLO/黃金集直接吃。"""
    df = load_defects()
    out: dict[str, list[tuple[str, list[int]]]] = {}
    for _, r in df.iterrows():
        if not r.box:
            continue
        try:
            v = json.loads(r.box)
        except Exception:
            continue
        for one in v if isinstance(v, list) else []:
            if len(one) == 4:
                out.setdefault(r.defectType, []).append((r.fileId, [int(c) for c in one]))
    return out
