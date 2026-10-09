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

from core import paths
from core.defects import load_defects

IOU_MATCH = 0.5  # ROADMAP #7：框 IoU ≥ 0.5 且同類＝一致


def load_trade(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"fileId", "cls"} <= set(df.columns):
        raise ValueError(f"{path} 需要 fileId,cls 兩欄")
    return df.set_index("fileId")


def _trade_labels(df: pd.DataFrame, name: str) -> pd.Series:
    if "cls" not in df:
        raise ValueError(f"{name} 需要 cls 欄與 fileId index")
    if any(not isinstance(f, str) or not f.strip() for f in df.index):
        raise ValueError(f"{name} 的 fileId 不可空白")
    if not df.index.is_unique:
        raise ValueError(f"{name} 的 fileId 重複，請先確認標註版本")
    return df.cls.fillna("").astype(str).str.strip()


def trade_gate(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """每類 n = 任一人選該類的照片聯集；一致、分歧、單邊各算一次。

    ceiling 的分母是兩份名單的照片聯集，缺標也占分母；pair 另記雙方都有答案的張數。
    雙方都未列出的照片需由黃金集 manifest 完整性檢查補上，不在此函式推測。
    """
    la, lb = _trade_labels(a, "A"), _trade_labels(b, "B")
    ids = sorted(set(la.index) | set(lb.index))
    rows, per_class = [], {}
    agree_n = pair_n = one_sided = unlabeled = 0
    for f in ids:
        ca, cb = la.get(f, ""), lb.get(f, "")
        ok = bool(ca and cb and ca == cb)
        agree_n += ok
        if ca and cb:
            pair_n += 1
            kind = "一致" if ok else "分歧"
        elif ca or cb:
            one_sided += 1
            kind = "A 有 B 無" if ca else "B 有 A 無"
        else:
            unlabeled += 1
            kind = "雙方未標"
        rows.append({"fileId": f, "A": ca, "B": cb, "agree": ok, "kind": kind})
        for c in sorted({ca, cb} - {""}):
            s = per_class.setdefault(c, {"n": 0, "agree": 0})
            s["n"] += 1
            s["agree"] += ok
    rep = pd.DataFrame(
        [
            {
                "cls": c,
                "n": s["n"],
                "一致率": round(s["agree"] / s["n"], 3),
                "本輪可評": "✅" if s["n"] >= 30 else f"n={s['n']} <30",
            }
            for c, s in sorted(per_class.items(), key=lambda x: -x[1]["n"])
        ],
        columns=["cls", "n", "一致率", "本輪可評"],
    )
    summary = {
        "total": len(ids),
        "pair": pair_n,
        "agree": agree_n,
        "one_sided": one_sided,
        "unlabeled": unlabeled,
        "ceiling": round(agree_n / len(ids), 3) if ids else None,
    }
    return pd.DataFrame(rows, columns=["fileId", "A", "B", "agree", "kind"]), {
        "per_class": rep,
        "summary": summary,
    }


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
    """每樣態 n = 涉及該樣態的配對或單邊框事件數；一個一致配對只算一次。"""
    from explain import iou  # 與 Gemini evidence / review.csv box 同一份 IoU 定義

    if not src_a or not src_b or src_a == src_b:
        raise ValueError("需要兩個不同且非空白的標註來源")
    if defects.empty:
        defects = defects.reindex(columns=["fileId", "source", "defectType", "box"])
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
                for t in sorted({ta, best}):
                    s = per_pattern.setdefault(t, {"n": 0, "agree": 0, "paired": 0, "one_sided": 0})
                    s["n"] += 1
                    s["paired"] += 1
                    s["agree"] += ok
                rows.append({"fileId": f, "kind": kind, "A": ta, "B": best, "iou": round(best_iou, 3)})
            else:
                stats["one_sided"] += 1
                s = per_pattern.setdefault(ta, {"n": 0, "agree": 0, "paired": 0, "one_sided": 0})
                s["n"] += 1
                s["one_sided"] += 1
                rows.append({"fileId": f, "kind": "A 有 B 無", "A": ta, "B": "", "iou": 0.0})
        for j, (tb, _bb) in enumerate(lb):
            if j not in used_b:
                stats["one_sided"] += 1
                s = per_pattern.setdefault(tb, {"n": 0, "agree": 0, "paired": 0, "one_sided": 0})
                s["n"] += 1
                s["one_sided"] += 1
                rows.append({"fileId": f, "kind": "B 有 A 無", "A": "", "B": tb, "iou": 0.0})
    rep = pd.DataFrame(
        [
            {
                "樣態": t,
                "n": s["n"],
                "配對框數": s["paired"],
                "單邊框數": s["one_sided"],
                "一致率": round(s["agree"] / s["n"], 3),
                "本輪可評": "✅" if s["n"] >= 30 else f"n={s['n']} <30，本輪不評",
            }
            for t, s in sorted(per_pattern.items(), key=lambda x: -x[1]["n"])
        ],
        columns=["樣態", "n", "配對框數", "單邊框數", "一致率", "本輪可評"],
    )
    return pd.DataFrame(rows, columns=["fileId", "kind", "A", "B", "iou"]), {
        "per_pattern": rep,
        "summary": stats,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trade-a", default="")
    ap.add_argument("--trade-b", default="")
    ap.add_argument("--defect-sources", default="", help="defects.csv 的兩個 source，如 g1-A,g1-B")
    a = ap.parse_args()
    if bool(a.trade_a) != bool(a.trade_b):
        ap.error("--trade-a 與 --trade-b 必須一起提供")

    out_rows: list[pd.DataFrame] = []
    total_items = 0
    if a.trade_a and a.trade_b:
        df, rep = trade_gate(load_trade(a.trade_a), load_trade(a.trade_b))
        total_items += len(df)
        out_rows.append(df[df.kind != "一致"])
        s = rep["summary"]
        print(
            f"== trade 一致度：{s['ceiling']}（{s['agree']}/{s['total']}，"
            f"雙方有答案 {s['pair']}，單邊缺標 {s['one_sided']}，雙方未標 {s['unlabeled']}）"
        )
        print(rep["per_class"].to_string(index=False))
    if a.defect_sources:
        sa, sb = a.defect_sources.split(",")
        df, rep = defect_gate(load_defects(), sa.strip(), sb.strip())
        total_items += len(df)
        out_rows.append(df[df.kind != "一致"])
        s = rep["summary"]
        print(
            f"== defect 框配對：一致 {s['agree']} / 配對 {s['pair']}，樣態歧義 {s['type_conflict']}，單邊 {s['one_sided']}"
        )
        print(rep["per_pattern"].to_string(index=False))
    if not out_rows:
        print("沒有輸入。--trade-a/--trade-b 或 --defect-sources 至少一組。")
        return 1
    if not total_items:
        print("沒有可比較的標註，不產生仲裁報告。")
        return 1
    arbit = pd.concat(out_rows, ignore_index=True)
    out = paths.GOLDEN / "gate_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    arbit.to_csv(out, index=False)
    print(f"\n仲裁清單 {len(arbit)} 筆 → {out}（主任裁終審，黃金答案以此為準）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
