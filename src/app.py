"""操作台。主導覽四格：收件匣／總覽／報告／進階（QMS 稽核與規範庫在「進階」裡）。

PMS 與 QMS 的資料、標籤、模型全部各自獨立（實測混訓有害，見 README）；QMS 與規範庫
只是收進「進階」，不是併進 PMS 的流程。

各系統的分頁與共用元件拆在 src/ui/：
    common.py        CSS、nav()、確認者欄、小工具（多分頁共用）
    inbox.py         收件匣（每日分流後要人看的照片）＋總覽頁的排程狀態
    data.py          快取載入器（manifest / QMS / 本地預測）
    pipeline.py      重跑管線 + 判斷依據（遮擋法）
    review_ui.py     複核佇列 + 標框畫布
    legacy_ui.py     歷史資料（舊 pptx）
    report_view.py   報告格式（PMS / QMS 共用）
    pms_page.py      PMS 分頁路由
    pms_workbench.py 照片工種分類、新工種候選與審閱包
    qms_page.py      QMS 四分頁
    standards_page.py 規範庫五分頁

分頁導航用 nav()（radio）不用 st.tabs：st.tabs 的選取是純前端狀態，元件樹一變
（例如標框畫布出現）就重置回第一頁——標一次框就被彈走一次。radio 的值存在
session_state，任何 rerun 都不會掉。

    uv run streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

from ui.common import inject_css, nav
from ui.inbox import inbox
from ui.pms_page import pms_page
from ui.qms_page import qms_page
from ui.standards_page import standards_page

st.set_page_config(page_title="eagle-vision", layout="wide", page_icon="🦅")
inject_css()

# ---------- 進入點 -----------------------------------------------------
# 主導覽只放每天會用的三格（2026-10-09 收斂）：收件匣＝今天要人看的、總覽＝系統有沒有在跑、
# 報告＝模型考得怎樣。其餘工具（手動同步重訓、逐張工作台、舊資料、規則編輯、QMS、規範庫）收進「進階」。
SECTION = st.segmented_control(
    "區塊", ["收件匣", "總覽", "報告", "進階"], default="收件匣", key="main_nav", label_visibility="collapsed"
)
ADVANCED = ["同步與重訓", "照片工種", "進階複核", "新工種候選", "歷史資料", "標籤規則", "QMS 稽核", "規範庫"]

if SECTION == "總覽":
    pms_page("總覽")
elif SECTION == "報告":
    pms_page("報告")
elif SECTION == "進階":
    pick = ADVANCED[nav(ADVANCED, key="adv_nav").index(True)]
    if pick == "QMS 稽核":
        qms_page(nav(["① 同步", "② 資料總覽", "③ 照片", "④ 報告"], key="qms_nav"))
    elif pick == "規範庫":
        standards_page(
            nav(["① QS 標準", "② 合約工作約定", "③ 衝突比對", "④ 工種介面", "⑤ 合約相依缺口"], key="std_nav")
        )
    else:
        pms_page(pick)
else:  # segmented_control 被取消選取時回傳 None，視同收件匣
    inbox()
