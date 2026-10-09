#!/usr/bin/env python
"""QS 標準 + 檢查項 → Excel（給同事查閱用）。

純讀本地 `reference/iso/`（catalog.yaml 目錄 + raw/*.tsv 檢查項），不碰 QMS API。

    uv run src/qs_excel.py                  # 寫到 reports/QS標準總表.xlsx
    uv run src/qs_excel.py -o /tmp/a.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from core import paths, qs_data

CATALOG = paths.ROOT / "reference" / "iso" / "catalog.yaml"
OUT = paths.ROOT / "reports" / "QS標準總表.xlsx"

KIND_LABEL = {t: f"{t} {desc}" for t, desc, _ in qs_data.RULES} | {"A": "A 純視覺"}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    cat = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    cats: dict[str, str] = cat["categories"]
    docs = qs_data.load()

    def cat_of(doc_no: str) -> str:
        code = doc_no[2:4]
        return f"{code} {cats.get(code, '?')}"

    # 標準目錄：以 catalog.yaml 為母體（含 raw 還沒收的 8 份，檢查項數 = 0 看得出缺口）
    rows = []
    for d in cat["docs"]:
        doc = docs.get(d["docNo"])
        rows.append(
            {
                "大類": cat_of(d["docNo"]),
                "docNo": d["docNo"],
                "標準名稱": d["name"],
                "版次": d["version"],
                "檢查項數": len(doc.required) if doc else 0,
                "工序階段數": len(doc.phases) if doc else 0,
                "工序型態": doc.shape if doc and doc.items else "—",
                # #DOC 列有、底下 item 卻 0 筆 = 標準存在但 checklist 還沒謄打（README 的「已知缺口」）
                "原始資料": "有" if doc and doc.items else "缺 checklist item",
                "isoInfoId": d["id"],
            }
        )
    # raw 有但 catalog 沒有的（如消歧碼 QS0907B 若未登錄）也要現形，不能靜默掉
    known = {d["docNo"] for d in cat["docs"]}
    for doc_no, doc in docs.items():
        if doc_no not in known:
            rows.append(
                {
                    "大類": cat_of(doc_no),
                    "docNo": doc_no,
                    "標準名稱": doc.name,
                    "版次": "?",
                    "檢查項數": len(doc.required),
                    "工序階段數": len(doc.phases),
                    "工序型態": doc.shape,
                    "原始資料": "有（catalog.yaml 未登錄）",
                    "isoInfoId": doc.iso_info_id,
                }
            )
    index = pd.DataFrame(rows).sort_values("docNo", ignore_index=True)

    items = pd.DataFrame(
        [
            {
                "大類": cat_of(i.doc_no),
                "docNo": i.doc_no,
                "標準名稱": docs[i.doc_no].name,
                "項次": i.item_no,
                "主鍵": i.key,
                "層級": i.depth,
                "類型": "階段節點" if i.status == "O" else "檢查項",
                "內容": i.name,
                "判定工具": KIND_LABEL.get(i.kind, i.kind),
                "請款靶": "Y" if i.is_billing else "",
                "罰則": "Y" if i.is_penalty else "",
                "合約相依": "Y" if i.is_contract else "",
            }
            # docNo 排序後，項次照 raw 原始順序（raw 本身未排序，保留原貌）
            for doc_no in sorted(docs)
            for i in docs[doc_no].items
        ]
    )
    return index, items


def write(out: Path, index: pd.DataFrame, items: pd.DataFrame) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # ponytail: 欄寬用字元數粗估（CJK 算 2 格），不量實際字型寬度——夠看就好
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for sheet, df in (("標準目錄", index), ("檢查項", items)):
            df.to_excel(xw, sheet_name=sheet, index=False)
            ws = xw.sheets[sheet]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for n, col in enumerate(df.columns, 1):
                w = max(
                    len(str(col)) * 2,
                    *(sum(2 if ord(c) > 0x2E80 else 1 for c in str(v)) for v in df[col].head(200)),
                )
                ws.column_dimensions[ws.cell(1, n).column_letter].width = min(w + 2, 70)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    index, items = build()
    assert len(items) == len(qs_data.all_items()), "檢查項列數與 qs_data 不一致——raw 解析掉東西了"
    write(a.out, index, items)
    print(f"{a.out}")
    print(
        f"  標準目錄 {len(index)} 份（{(index['原始資料'].str.startswith('缺')).sum()} 份無 checklist item）"
    )
    print(
        f"  檢查項 {len(items)} 列（檢查項 {(items['類型'] == '檢查項').sum()} / 階段節點 {(items['類型'] == '階段節點').sum()}）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
