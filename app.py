# app.py (V2.0 Presentation Shell — 已接上真實服務)
"""V2.0 展示層。所有計算經 pipeline/ 協調器與 core/ 服務層，本頁不含業務邏輯。

與 src/app.py（現行操作台，功能最全）的分工：這裡是架構文件承諾的
「單點入口 + 服務分層」示範殼——展示層薄、 orchestrator 調度、服務層算。
    uv run streamlit run app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline.run_full_qc_workflow import (  # noqa: E402
    get_workflow_status,
    run_full_qc_workflow,
)

st.set_page_config(page_title="eagle-vision", layout="wide", page_icon="🦅")
st.title("🦅 Eagle Vision (QC Field Report System)")
st.caption("V2.0 shell · 計算一律走 pipeline → core 服務層；完整操作台在 `src/app.py`")

tabs = st.tabs(["① Snapshot", "② Sync", "③ Review Queue", "④ Reports"])

with tabs[0]:
    st.subheader("System Snapshot（唯讀）")
    if st.button("載入快取外的全新快照", type="primary"):
        with st.spinner("QS + 合約 + 標籤…"):
            r = run_full_qc_workflow(stage="snapshot")
        st.write(f"status: `{r['status']}` · QS {r.get('qs_docs')} 份 · "
                 f"合約 {r.get('contracts')} 份 · 標籤快照 {r.get('labeled')} 張")
        st.dataframe(r["details"], hide_index=True, width="stretch")

with tabs[1]:
    st.subheader("Data Sync（抓新日報，會動 data/）")
    if st.button("執行 sync（src/sync.py）", type="primary"):
        with st.spinner("同步中…"):
            r = run_full_qc_workflow(stage="sync")
        st.write(f"status: `{r['status']}`")
        st.dataframe(r["details"], hide_index=True, width="stretch")

with tabs[2]:
    st.subheader("Review Queue（真實分層，待人工裁決）")
    q = get_workflow_status("queue_preview")
    if len(q):
        st.dataframe(q[["tier", "cls", "mPred", "why", "reportDate"]].head(20),
                     hide_index=True, width="stretch")
        st.caption("裁決入口在現行操作台：`make app` → ④ 複核佇列")
    else:
        st.success("佇列是空的。")

with tabs[3]:
    st.subheader("Reports（歷次評估）")
    rows = get_workflow_status("report_status")
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("詳細分數：reports/JOURNAL.md（`make journal`）")
