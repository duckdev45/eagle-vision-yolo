"""待人看的照片清單（CLI）。清單本身由每日分流算好（core/routing.py），這裡只印——
收件匣、工作台、`make pms-status` 與本 CLI 讀的是同一份，數字不會各說各話。

    uv run src/review.py                      # 原因統計＋前 30 張（make queue）
    uv run src/review.py --reason 標題沒有對應規則   # 只看某個原因（舊 --orphans＝規則沒接住的）
    uv run src/review.py --since 2026-08-19   # 只看某天之後的日報

分流結果是最近一次 `make daily`／`make route` 的；要最新就先跑 `make route`。
"""

from __future__ import annotations

import argparse

from core import routing
from core.labeler import orphan_reviews
from core.pms_review import load_pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default="", help=f"只看某個原因：{'、'.join(routing.REASONS)}")
    ap.add_argument("--since", default="", help="只看 reportDate >= 這天的")
    ap.add_argument("--limit", type=int, default=30)
    a = ap.parse_args(argv)

    q = routing.queue_items()
    _, meta = routing.latest()
    if not meta:
        print("還沒有分流結果：先跑 `make route`。")
        return 1
    print(f"分流 {meta.get('routedAt', '?')} · 模型 {meta.get('model', '?')} · 待看 {len(q)} 張")
    if orphan := orphan_reviews():
        print(f"⚠ {len(orphan)} 筆裁決指到已不存在的類別 {sorted(set(orphan.values()))}——重裁補上新名字")
    if a.reason:
        q = q[q.reason.str.contains(a.reason, regex=False)]
    if a.since:
        q = q[q.reportDate.fillna("") >= a.since]
    for w in routing.REASONS:
        n = int(q.reason.str.contains(w, regex=False).sum())
        if n:
            print(f"  {w}：{n} 張")
    if q.empty:
        print("沒有符合條件的照片。")
        return 0
    pool = load_pool()
    titles = dict(zip(pool.fileId, pool.title))
    print(f"\n前 {min(a.limit, len(q))} 張（越前面越該先看）：")
    print(f"{'fileId':10} {'reportDate':11} {'規則':14} {'模型':14} 原因｜標題")
    for r in q.head(a.limit).itertuples():
        reason = "例行確認" if r.bucket == "audit" else r.reason  # 抽查照不露底，同收件匣
        print(
            f"{str(r.fileId)[:8]:10} {r.reportDate!s:11} {r.ruleClass or '—':14} {r.modelClass or '—':14}"
            f" {reason}｜{titles.get(r.fileId, '')}"
        )
    print("\n裁決在 `make pms-app` 的收件匣（要畫證據框到「進階 → 進階複核」）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
