"""G1 gate：兩位標註者的一致度（人類天花板）＋模糊仲裁清單。

標註回來之後跑（標前先看 src/g1_sample.py 的 README）：
  trade  兩人各一份 labels-{名字}.csv（fileId,cls）→ 同類=一致、不同類=分歧
  defect defects.csv 裡 source=g1-A / g1-B 的框 → 框配對：
         IoU ≥ 0.5 且同樣態 = 一致；同位置不同樣態 = 樣態歧義；
         一人有一人無 = 漏標/多標 → 全進仲裁清單，由主任（CVAT 微調者）裁終審。

per-class 報告遵守 WORKFLOW §5 紀律：樣本 <30 的樣態「本輪不評」（gateEligible=0），
recall 要下結論 ≥50。

    uv run src/g1_gate.py --trade-a labels-A.csv --trade-b labels-B.csv \
        --defect-sources g1-A,g1-B

產出：data/golden/gate_report.csv（分歧/模糊清單，主任仲裁用）＋ console 報告。
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

import paths
from core.defects import load_defects

IOU_MATCH = 0.5  # ROADMAP #7：框 IoU ≥ 0.5 且同類＝一致


def load_trade(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    assert {"fileId", "cls"} <= set(df.columns), f"{path} 需要 fileId,cls 兩欄"
    return df.set_index("fileId")


def trade_gate(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    common = a.index.intersection(b.index)
    rows, per_class = [], {}
    agree_n = 0
    for f in common:
        ca, cb = a.loc[f, "cls"], b.loc[f, "cls"]
        ok = ca == cb
        agree_n += ok
        rows.append({"fileId": f, "A": ca, "B": cb, "agree": ok, "kind": "一致" if ok else "分歧"})
        for c in (ca, cb):
            s = per_class.setdefault(c, {"n": 0, "agree": 0})
            s["n"] += 1
        if ok:
            s = per_class[ca]
            s["agree"] += 1
    rep = pd.DataFrame(
        [
            {
                "cls": c,
                "n": s["n"],
                "一致率": round(s["agree"] / s["n"], 3),
                "本輪可評": "✅" if s["n"] >= 30 else f"n={s['n']} <30",
            }
            for c, s in sorted(per_class.items(), key=lambda x: -x[1]["n"])
        ]
    )
    summary = {
        "pair": len(common),
        "agree": agree_n,
        "ceiling": round(agree_n / len(common), 3) if len(common) else None,
    }
    return pd.DataFrame(rows), {"per_class": rep, "summary": summary}


def _boxes(rows: pd.DataFrame) -> dict[str, list[tuple[str, list[int]]]]:
    out: dict[str, list[tuple[str, list[int]]]] = {}
    for _, r in rows.iterrows():
        if not r.box:
            continue
        try:
            v = json.loads(r.box)
        except Exception:
            continue
        for one in v if isinstance(v, list) else []:
            if len(one) == 4:
                out.setdefault(r.fileId, []).append((r.defectType, [int(c) for c in one]))
    return out


def defect_gate(defects: pd.DataFrame, src_a: str, src_b: str) -> tuple[pd.DataFrame, dict]:
    from explain import iou  # 與 Gemini evidence / review.csv box 同一份 IoU 定義

    A, B = _boxes(defects[defects.source == src_a]), _boxes(defects[defects.source == src_b])
    rows, per_pattern = [], {}
    stats = {"pair": 0, "agree": 0, "type_conflict": 0, "one_sided": 0}
    for f in sorted(set(A) | set(B)):
        la, lb = A.get(f, []), B.get(f, [])
        used_b: set[int] = set()
        for ta, ba in la:
            best, best_j, best_iou = None, None, 0.0
            for j, (tb, bb) in enumerate(lb):
                if j in used_b:
                    continue
                v = iou(ba, bb)
                if v > best_iou:
                    best, best_j, best_iou = tb, j, v
            if best is not None and best_iou >= IOU_MATCH:
                used_b.add(best_j)
                stats["pair"] += 1
                ok = ta == best
                stats["agree"] += ok
                stats["type_conflict"] += not ok
                kind = "一致" if ok else "樣態歧義"
                for t in (ta, best):
                    s = per_pattern.setdefault(t, {"n": 0, "agree": 0})
                    s["n"] += 1
                if ok:
                    per_pattern[ta]["agree"] += 1
                rows.append({"fileId": f, "kind": kind, "A": ta, "B": best, "iou": round(best_iou, 3)})
            else:
                stats["one_sided"] += 1
                rows.append({"fileId": f, "kind": "A 有 B 無", "A": ta, "B": "", "iou": 0.0})
        for j, (tb, _bb) in enumerate(lb):
            if j not in used_b:
                stats["one_sided"] += 1
                rows.append({"fileId": f, "kind": "B 有 A 無", "A": "", "B": tb, "iou": 0.0})
    rep = pd.DataFrame(
        [
            {
                "樣態": t,
                "配對框數": s["n"],
                "一致率": round(s["agree"] / s["n"], 3),
                "本輪可評": "✅" if s["n"] >= 30 else f"n={s['n']} <30，本輪不評",
            }
            for t, s in sorted(per_pattern.items(), key=lambda x: -x[1]["n"])
        ]
    )
    return pd.DataFrame(rows), {"per_pattern": rep, "summary": stats}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trade-a", default="")
    ap.add_argument("--trade-b", default="")
    ap.add_argument("--defect-sources", default="", help="defects.csv 的兩個 source，如 g1-A,g1-B")
    a = ap.parse_args()

    out_rows: list[pd.DataFrame] = []
    if a.trade_a and a.trade_b:
        df, rep = trade_gate(load_trade(a.trade_a), load_trade(a.trade_b))
        out_rows.append(df[df.kind != "一致"])
        print(
            f"== trade 一致度（人類天花板）：{rep['summary']['ceiling']}（{rep['summary']['agree']}/{rep['summary']['pair']}）"
        )
        print(rep["per_class"].to_string(index=False))
    if a.defect_sources:
        sa, sb = a.defect_sources.split(",")
        df, rep = defect_gate(load_defects(), sa.strip(), sb.strip())
        out_rows.append(df[df.kind != "一致"])
        s = rep["summary"]
        print(
            f"== defect 框配對：一致 {s['agree']} / 配對 {s['pair']}，樣態歧義 {s['type_conflict']}，單邊 {s['one_sided']}"
        )
        print(rep["per_pattern"].to_string(index=False))
    if not out_rows:
        print("沒有輸入。--trade-a/--trade-b 或 --defect-sources 至少一組。")
        return 1
    arbit = pd.concat(out_rows, ignore_index=True)
    out = paths.GOLDEN / "gate_report.csv"
    arbit.to_csv(out, index=False)
    print(f"\n仲裁清單 {len(arbit)} 筆 → {out}（主任裁終審，黃金答案以此為準）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
