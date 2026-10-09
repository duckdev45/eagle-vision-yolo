"""PMS 日報系統頁：依名稱分派到各區塊（收件匣在 inbox.py）。"""

from __future__ import annotations

import os

import streamlit as st

import organize
import sync as sync_mod
from core import paths
from labels import Labeler

from .common import run_step
from .data import load_manifest
from .inbox import status_panel
from .pipeline import pipeline_panel
from .pms_workbench import workbench
from .report_view import report_view
from .review_ui import review_queue


def pms_page(section: str) -> None:
    if section == "同步與重訓":
        st.subheader("每日同步")
        st.caption(
            f"來源 {os.getenv('API_BASE_URL', '（未設 API_BASE_URL）')} · raw 只抄不改 · derived 隨時可砍"
        )

        # 這頁只放三顆按鈕，各自負責一件別人不做的事。
        # 之前還有一顆「前處理（遮蔽 + 縮圖）」，它是「重跑模型」第 2 步的子集、
        # 沒有任何自己的選項，連說明都寫著「整條鏈看下面」——那不是按鈕，是陷阱：
        # 按了以為有進度，其實下面還得整條再跑一次。已移除，要單步跑用 make。
        with st.container(border=True):
            st.markdown("**1. 抓新資料** — 只抓，不重訓")
            st.caption(
                "想知道今天有沒有新照片就按這顆。**抓完模型還是舊的**，"
                "要讓新照片進到模型得再跑下面的「全鏈重跑」。"
            )
            c1, c2 = st.columns([1, 2])
            limit = c1.number_input("只抓前 N 份日報（0 = 全部）", 0, 5000, 0, step=10)
            full = c1.checkbox(
                "忽略快取全部重抓",
                value=False,
                help="平常不用。SUBMITTED 且 version 沒變的日報預設不重抓 detail。",
            )
            if c2.button("▶ 同步日報 + 下載照片", type="primary"):
                run_step(sync_mod.sync, full=full, limit=int(limit) or None)

        with st.container(border=True):
            pipeline_panel()

        with st.container(border=True):
            st.markdown("**3. 建照片結構樹** — 跟訓練無關")
            st.caption(
                f"重建 `{paths.TREE}` 的 symlink（by-date / by-site / by-class），"
                "只影響用檔案總管翻照片。不在上面任何一條鏈裡，想跑再跑。"
            )
            if st.button("▶ 建照片結構樹"):
                run_step(organize.build)

        st.divider()
        m = load_manifest()
        a, b, c, d = st.columns(4)
        a.metric("manifest 列數", len(m))
        b.metric("raw 照片", len(list(paths.PHOTOS.glob("*"))) if paths.PHOTOS.exists() else 0)
        c.metric("已前處理", len(list(paths.IMAGES.glob("*.jpg"))) if paths.IMAGES.exists() else 0)
        d.metric(
            "日報快照", len(list(paths.REPORTS_JSON.glob("*.json"))) if paths.REPORTS_JSON.exists() else 0
        )

    if section == "總覽":
        status_panel()
        workbench("overview")

    if section == "照片工種":
        workbench("photos")

    if section == "進階複核":
        review_queue()

    if section == "新工種候選":
        workbench("candidates")

    if section == "標籤規則":
        st.subheader("labels.yaml")
        st.caption(
            "順序即優先權，第一個命中者勝。這裡存檔不會跑 tests/test_labels_yaml.py 的順序契約——"
            "正式改規則請在 repo 裡改並跑 make test。"
        )
        text = st.text_area("規則", paths.LABELS_YAML.read_text(), height=320)
        if st.button("💾 存檔並重算"):
            paths.LABELS_YAML.write_text(text)
            st.cache_data.clear()
            st.rerun()
        probe = st.text_input("試打一個標題", "13F外牆打底粉光")
        if probe:
            st.write("→", Labeler.load().label(probe) or "（排除）")
        st.caption("完整類別與樣本門檻見「總覽」；未知照片群組及新類核准見「進階 → 新工種候選」。")

    if section == "報告":
        runs = (
            sorted(
                [
                    p
                    for p in paths.REPORTS_OUT.glob("*")
                    if p.is_dir() and (p / "config.json").exists() and (p / "metrics.json").exists()
                ],
                reverse=True,
            )
            if paths.REPORTS_OUT.exists()
            else []
        )
        report_view(runs, "pms_run")
