"""操作台。最上層先選系統：PMS 日報 / QMS 稽核 / 規範庫。

兩邊的資料、標籤、模型全部各自獨立（實測混訓有害，見 README），所以在最上層就
分開，而不是塞成同一排分頁裡的一格——那會讓人以為它們是同一件事的兩個面向。

各系統的分頁與共用元件拆在 src/ui/：
    common.py        CSS、nav()、小工具（多分頁共用）
    data.py          快取載入器（manifest / QMS / 本地預測）
    pipeline.py      重跑管線 + 判斷依據（遮擋法）
    review_ui.py     複核佇列 + 標框畫布
    legacy_ui.py     歷史資料（舊 pptx）
    report_view.py   報告格式（PMS / QMS 共用）
    pms_page.py      PMS 七分頁
    qms_page.py      QMS 四分頁
    standards_page.py 規範庫五分頁

    uv run streamlit run src/app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from ui.common import inject_css  # noqa: E402
from ui.pms_page import pms_page  # noqa: E402
from ui.qms_page import qms_page  # noqa: E402
from ui.standards_page import standards_page  # noqa: E402

st.set_page_config(page_title="eagle-vision", layout="wide", page_icon="🦅")
inject_css()

# ---------- 進入點 -----------------------------------------------------
SYSTEM = st.segmented_control(
    "系統", ["PMS 日報", "QMS 稽核", "規範庫"], default="PMS 日報",
    label_visibility="collapsed")

if SYSTEM == "QMS 稽核":
    with st.container():
        qms_tabs = st.tabs(["① 同步", "② 分佈", "③ 照片", "④ 報告", "⑤ 模型"])
        qms_page(qms_tabs)
elif SYSTEM == "規範庫":
    with st.container():
        std_tabs = st.tabs(["① QS 標準", "② 合約工作約定", "③ 衝突比對",
                            "④ 工種介面", "⑤ 合約相依缺口"])
        standards_page(std_tabs)
else:
    with st.container():
        pms_tabs = st.tabs(["① 同步", "② 分佈", "③ 照片", "④ 複核佇列",
                            "⑤ 歷史資料", "⑥ 標籤規則", "⑦ 報告"])
        pms_page(pms_tabs)
