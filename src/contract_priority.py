"""合約優先序重排（ROADMAP 待決 #4）：以 97 項合約相依 REQUIRED 重算收集順序。

舊排序基於 24 項（合約庫初版時的盤點）；現在 contract_items() 實測 97 項
REQUIRED 的判定基準指向合約（佔 9.0%）。RAG 單灌 QS 答不出這批——缺哪份
工明，哪個工種的 AI 判定就啞火。

排序邏輯：
  1. contract_items() 按 QS 文件計數，文件歸併成「工明族群」（QS03 結構、
     QS04 泥作、QS05 裝修、QS06 門窗、QS07 防水、QS09 機電…）
  2. 扣掉 mappings.yaml QS_ANSWERS 已映射的項目（5 項）
  3. 已有工明的族群（TRADE_TO_QS）與完全沒工明的族群分兩層排——
     沒工明的族群缺的是「整份合約」，有工明的缺的是「條款級映射」
  4. 未映射數降序 = 收集/映射的優先序

    uv run src/contract_priority.py        # 印表 + 寫 reference/contract/priority.tsv
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from core import paths
from qsdata import contract_items, load

# doc_no 前綴 → 工明族群。QS 編號的前兩碼就是工程大類（QS0301 模板、QS0402 泥作…）
FAMILY = {
    "QS03": "結構（模板/鋼筋/混凝土）",
    "QS04": "泥作/磁磚",
    "QS05": "裝修（木作/石材/塗裝…）",
    "QS06": "門窗/玻璃",
    "QS07": "防水",
    "QS08": "材料檢驗",
    "QS09": "機電（電氣/給排水/消防/電梯）",
    "QS10": "其他",
}


def family_of(doc_no: str) -> str:
    return FAMILY.get(doc_no[:4], FAMILY.get(doc_no[:4], "其他"))


def mappings() -> dict:
    p = paths.ROOT / "reference" / "contract" / "mappings.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(paths.ROOT / "reference" / "contract" / "priority.tsv"))
    a = ap.parse_args()

    docs = load()
    items = contract_items(docs)
    m = mappings()
    covered_keys = set((m.get("QS_ANSWERS") or {}).keys())
    trade_to_qs = m.get("TRADE_TO_QS") or {}
    covered_docs = {d for qs_list in trade_to_qs.values() for d in qs_list}

    rows = []
    for doc_no, cnt in Counter(i.doc_no for i in items).most_common():
        cov = sum(1 for i in items if i.doc_no == doc_no and i.key in covered_keys)
        fam = family_of(doc_no)
        has = doc_no in covered_docs
        rows.append(
            {
                "族群": fam,
                "QS文件": doc_no,
                "名稱": docs[doc_no].name if doc_no in docs else "?",
                "合約相依REQUIRED": cnt,
                "已映射條款": cov,
                "未映射": cnt - cov,
                "該工種已有工明": "✅" if has else "—",
            }
        )

    # 兩張榜單：沒工明的族群缺「整份合約」（收集問題）；有工明的缺「條款級映射」
    # （映射問題，便宜得多）。0 未映射的族群不上榜——它已經做完。
    fam_agg: dict[str, dict] = {}
    for r in rows:
        f = fam_agg.setdefault(r["族群"], {"未映射": 0, "總數": 0, "有工明": True})
        f["未映射"] += r["未映射"]
        f["總數"] += r["合約相依REQUIRED"]
        f["有工明"] = f["有工明"] and r["該工種已有工明"] == "✅"

    lines = [f"合約收集優先序（97 項合約相依 REQUIRED，{datetime.now().date()} 重排）", ""]
    missing = sorted(
        ((fam, a) for fam, a in fam_agg.items() if not a["有工明"] and a["未映射"] > 0),
        key=lambda kv: -kv[1]["未映射"],
    )
    need_map = sorted(
        ((fam, a) for fam, a in fam_agg.items() if a["有工明"] and a["未映射"] > 0),
        key=lambda kv: -kv[1]["未映射"],
    )

    lines.append("一、缺整份工明（收集問題）——按未映射項數排：")
    for i, (fam, agg) in enumerate(missing, 1):
        lines.append(f"  {i}. {fam}：未映射 {agg['未映射']} 項 / 合約相依 {agg['總數']} 項")
    if not missing:
        lines.append("  （無）")
    lines.append("")
    lines.append("二、已有工明、缺條款級映射（映射問題，便宜得多）：")
    for i, (fam, agg) in enumerate(need_map, 1):
        lines.append(f"  {i}. {fam}：未映射 {agg['未映射']} 項 / 合約相依 {agg['總數']} 項")
    if not need_map:
        lines.append("  （無——既有工明的族群條款映射已完成或本就無合約相依項）")
    lines += ["", "─" * 60, "族群內明細（依未映射數排序）：", ""]
    lines.append("族群\tQS文件\t名稱\t合約相依REQUIRED\t已映射\t未映射\t該工種已有工明")
    for r in sorted(rows, key=lambda x: (x["族群"], -x["未映射"])):
        lines.append(
            f"{r['族群']}\t{r['QS文件']}\t{r['名稱']}\t{r['合約相依REQUIRED']}\t"
            f"{r['已映射條款']}\t{r['未映射']}\t{r['該工種已有工明']}"
        )

    text = "\n".join(lines)
    print(text)
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
