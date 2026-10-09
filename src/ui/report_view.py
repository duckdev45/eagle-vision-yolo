"""報告分頁 —— 模型考得怎樣（讀 reports/ 的 metrics）。"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st


def report_view(runs, key: str):
    if not runs:
        st.info("還沒有報告。")
        return
    pick = st.selectbox("run", runs, format_func=lambda p: p.name, key=key)
    if (pick / "comparison.md").exists():
        st.markdown((pick / "comparison.md").read_text())
    m = json.loads((pick / "metrics.json").read_text()) if (pick / "metrics.json").exists() else {}
    if m:
        a, b, c = st.columns(3)
        a.metric("top-1", m["top1"])
        b.metric("macro-F1", m["macroF1"])
        c.metric("測試張數", m["support"])
        if m.get("coverageCurve"):
            st.caption("信心門檻 → 自動處理比例與該批準確率（決定何時能少靠人工）")
            st.dataframe(
                pd.DataFrame(m["coverageCurve"]).rename(
                    columns={
                        "threshold": "門檻",
                        "coverage": "自動處理",
                        "accuracyOnCovered": "該批準確率",
                        "n": "張數",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        if m.get("perClass"):
            st.caption("每類指標")
            st.dataframe(
                pd.DataFrame(m["perClass"]).T.sort_values("support", ascending=False),
                width="stretch",
                height=420,
            )
    if (pick / "confusion.png").exists():
        st.image(str(pick / "confusion.png"))
    errs = sorted((pick / "errors").glob("*.jpg")) if (pick / "errors").exists() else []
    if errs:
        st.caption("誤判樣本（檔名 = 真實_預測）")
        cols = st.columns(6)
        for i, e in enumerate(errs):
            cols[i % 6].image(str(e), caption=e.stem[:28], width="stretch")
