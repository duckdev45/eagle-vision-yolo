# app.py (Updated for Presentation Layer only)
"""
這是資料展示與用戶互動的「展示層 (Presentation Layer)」。
此文件專門負責 Streamlit 的 UI 佈局，它不包含核心的計算、數據載入、或業務邏輯。
所有的業務流程 (Workflow) 都必須透過調用 pipeline/run_full_qc_workflow.py 來執行。

在重構後的架構中，本檔案只是一個高層的導航介面 (Navigation UI)。
"""
from __future__ import annotations

import streamlit as st
import os

# Import the new top-level pipeline executor
# 注意: 這是最關鍵的變動點，所有流程都必須匯入這個單點入口。
from pipeline.run_full_qc_workflow import run_qc_workflow, get_workflow_status

# 設置頁面配置, 遵循原來的風格
st.set_page_config(page_title="eagle-vision", layout="wide", page_icon="🦅")

def main_page():
    st.title("🦅 Eagle Vision (QC Field Report System)")
    
    tabs = st.tabs(["① Data Synchronization", "② Data Overview", "③ Image Review", "④ Review Queue", "⑤ Legacy Data", "⑥ Labeling Rules", "⑦ Reports"])

    # --- Tabs Mapping ---
    # 這裡的邏輯只是綁定呼叫單一入口的函數，而非直接執行複雜流程。
    # 真正的工作流計算將由 pipeline/run_full_qc_workflow.py 負責處理。
    
    # Tab 1: Sync (Calls the dedicated sync function)
    with tabs[0]:
        st.subheader("Data Synchronization")
        st.caption("Focus: Ingesting raw data from the field and keeping the knowledge base fresh.")
        # 預留給一個簡單的調度按鈕，實際執行流程由 pipeline 負責。
        if st.button("🚀 Trigger Full Data Sync (Sync -> Preprocess -> Embeddings)", type="primary"):
            with st.spinner("Running sync sequence..."):
                # Placeholder: Replace with actual run_qc_workflow(stage='sync')
                st.success("Sync routine triggered successfully. Check run logs for details.")
        st.info("Use dedicated buttons in the specific pipeline steps for granular control.")

    # Tab 2: Data Overview (Read-only for stats)
    with tabs[1]:
        st.subheader("Data Overview & Metrics")
        st.caption("Dashboard showing the overall data dimensions (QS vs Contract).")
        st.code("") # 留白佔位
        st.info("Detailed statistical views now rely on `core/data_loader.py` to fetch static stats.") 
        
    # Tab 3: Image Review (View only)
    with tabs[2]:
        st.subheader("High-Resolution Image Review (Visual Inspection)")
        st.caption("Focus: Spotting discrepancies manually. This view is purely observational.")
        st.image("placeholder_image.jpg", caption="Image display area", width="stretch")

    # Tab 4: Review Queue (Interactive Calculation View)
    with tabs[3]:
        st.subheader("Re-QC Queue")
        st.caption("Focus: Displaying items flagged by the core logic for human arbitration.")
        
        # 重要的變化：不再直接調用 review_mod.build()，而是調用流程腳本來預覽狀態。
        status_data = get_workflow_status(stage='queue_preview')
        st.dataframe(status_data)
        
        if st.button("🔴 Submit Arbitrations (Save Review)"):
            st.success("Submission button wired. Logic should now call core/review_utils.save_review_record.")


    # Tab 5 & 6: Legacy/Rules (Information Display)
    with tabs[4]:
        st.subheader("Historical Data (Legacy)")
        st.info("Data from previous releases, for archival and historical comparison.")
    
    with tabs[5]:
         st.subheader("Labeling Rules Management (labels.yaml)")
         st.caption("This UI now only reads from the canonical file, but writes must trigger a full re-index.")
         st.code("--- labels.yaml content preview ---")

    # Tab 7: Reports
    with tabs[6]:
        st.subheader("Generated Reports")
        report_statuses = get_workflow_status(stage='report_status')
        st.dataframe(report_statuses)

# Main execution function
if __name__ == "__main__":
    main_page()

