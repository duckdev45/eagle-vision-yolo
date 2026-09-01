"""複核佇列 CLI。分層規則的實體在 core/review_utils.py（服務層），
操作台 ui/review_ui.py 與本 CLI 共用同一份——規則抄兩份的話，畫面上
看到的佇列跟命令列印的會慢慢對不起來。

    uv run src/review.py                # 還沒裁的，照優先序印
    uv run src/review.py --since 2026-08-19   # 某天之後才進來的
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # src/

import paths  # noqa: E402
import split as split_mod  # noqa: E402
from core.review_utils import TIER_NAMES, build, scores  # noqa: E402
from labels import Labeler, human_refs, load_reviews  # noqa: E402

__all__ = ["TIER_NAMES", "build", "scores", "queue"]


def _labeled() -> tuple[pd.DataFrame, Labeler]:
    """manifest + 規則標籤 + 人寫的兩個參考答案。與 app.labeled() 同一套，
    但不經過 Streamlit 的 cache_data（那個裝飾器在無 st context 下會炸）。"""
    from sync import _truthy
    df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
    if "active" in df:
        df = df[_truthy(df.active)]
    lab = Labeler.load()
    out = lab.apply(df)
    return out.join(human_refs(out, lab)), lab


def queue(split_name: str = "", since: str = "") -> pd.DataFrame:
    df, lab = _labeled()
    sc, test_ids = scores("siglip", split_name or split_mod.current())
    q = build(df, lab, sc, test_ids)
    done = set(load_reviews())
    q = q[~q.fileId.isin(done)]
    if since:
        q = q[q.reportDate.fillna("") >= since]
    return q.sort_values(["tier", "syncedAt", "reportDate"], ascending=False, kind="stable")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="", help="用哪一版模型算「難分」，預設 CURRENT")
    ap.add_argument("--since", default="", help="只看 reportDate >= 這天的")
    ap.add_argument("--limit", type=int, default=30)
    a = ap.parse_args()

    q = queue(a.split, a.since)
    cur = a.split or split_mod.current()
    print(f"模型 {cur} · 已裁 {len(load_reviews())} 筆 · 待裁 {len(q)} 張"
          + (f"（reportDate >= {a.since}）" if a.since else ""))
    if not len(q):
        print("佇列是空的。")
        sys.exit(0)
    for t in (4, 3, 2, 1):
        n = int((q.tier == t).sum())
        if n:
            print(f"  第 {t} 層 {TIER_NAMES[t]}：{n} 張")
    print(f"\n前 {min(a.limit, len(q))} 張（最強訊號在前）：")
    print(f"{'fileId':10} {'層':>2} {'reportDate':11} {'現在的標籤':14} {'模型猜':14} 訊號")
    for r in q.head(a.limit).itertuples():
        print(f"{str(r.fileId)[:8]:10} {r.tier:>2} {str(r.reportDate):11} "
              f"{str(r.cls):14} {str(r.mPred):14} {r.why}")
    print("\n裁決要人做：`make app` → ④ 複核佇列。裁完 `make model SPLIT=vN` 才會進模型。")
