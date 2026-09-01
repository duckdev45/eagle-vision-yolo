"""QMS 稽核系統頁（① 同步 ② 資料總覽 ③ 照片 ④ 報告）。"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import paths
import prepare
import split as split_mod
from .common import badge, run_step, txt, verdict
from .data import load_qms
from .report_view import report_view


def qms_page(tabs):
    df = load_qms()

    if tabs[0]:
        st.subheader("QMS 稽核照同步")
        st.caption(f"來源 {os.getenv('QMS_API_BASE_URL', 'https://qms.example.invalid/api/qms')}"
                   f" · 正式環境，只讀 · 母體 36,653 張")
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.number_input("抽樣張數", 500, 40000, 12000, step=500)
            cap = st.number_input("每中類上限（0 = 純隨機）", 0, 5000, 600, step=100)
            if st.button("▶ 抽樣 + 下載", type="primary"):
                import qms
                run_step(qms.sample, n=int(n), per_class_cap=int(cap))
        with c2:
            if st.button("▶ 重掃母體（只 query）"):
                import qms
                run_step(qms.scan_cells)
            if st.button("▶ 前處理（遮左下浮水印）"):
                run_step(prepare.run, kind="qms")
        with c3:
            if st.button("▶ 重建切分（按格切）"):
                run_step(split_mod.build_qms, name="qms-v1")
            if st.button("▶ 重跑 ONNX 推論"):
                import predict
                run_step(predict.run, ckpt="backbone-qms", src="qms")

        st.divider()
        a, b, c, d = st.columns(4)
        a.metric("manifest 列數", len(df))
        b.metric("raw 照片",
                 len(list(paths.QMS_PHOTOS.glob("*"))) if paths.QMS_PHOTOS.exists() else 0)
        c.metric("已前處理",
                 len(list(paths.QMS_IMAGES.glob("*.jpg"))) if paths.QMS_IMAGES.exists() else 0)
        d.metric("格數", df.constructionInsId.nunique() if len(df) else 0)

    if not len(df):
        for t in tabs[1:]:
            with t:
                st.info("還沒有 QMS 資料。到「① 同步」抽樣下載，或跑 `uv run src/qms.py --sample 12000`")
        return

    has_pred = "pred" in df and df.pred.notna().any()

    if tabs[1]:
        st.subheader(f"{len(df)} 張 / {df.cls.nunique()} 個中類 / "
                     f"{df.constructionInsId.nunique()} 格")
        c1, c2 = st.columns(2)
        c1.caption("各中類張數")
        c1.bar_chart(df.cls.value_counts())
        c2.caption("建案 × 大類")
        c2.dataframe(pd.crosstab(df.l1, df.constrName), width="stretch")
        if has_pred:
            d = df[df.pred.notna()]
            st.caption("每類正確率（含訓練集，僅供概觀；正式數字看「④ 報告」）")
            # 用聚合不用 apply：apply(include_groups=False) 會把 cls 從 group 裡拿掉
            acc = (d.assign(hit=d.pred == d.cls).groupby("cls")
                   .agg(張數=("hit", "size"), 正確率=("hit", "mean"), 平均信心=("conf", "mean"))
                   .round(3).sort_values("正確率"))
            st.dataframe(acc, width="stretch", height=420)
        else:
            st.warning("尚無模型預測。到「① 同步」按「重跑 ONNX 推論」。")

    if tabs[2]:
        c1, c2, c3, c4 = st.columns(4)
        site = c1.selectbox("建案", ["全部"] + sorted(df.constrName.dropna().unique().tolist()),
                            key="q_site")
        d2 = df if site == "全部" else df[df.constrName == site]
        cls = c2.selectbox("中類", ["全部"] + sorted(d2.cls.unique().tolist()), key="q_cls")
        d2 = d2 if cls == "全部" else d2[d2.cls == cls]
        part = c3.selectbox("資料分割", ["全部", "test", "val", "train"], key="q_part")
        if part != "全部" and "part" in d2:
            d2 = d2[d2.part == part]
        view = c4.selectbox("看什麼", ["全部", "只看判錯的", "只看低信心"], key="q_view")
        if has_pred and view == "只看判錯的":
            d2 = d2[d2.pred.notna() & (d2.pred != d2.cls)]
        elif has_pred and view == "只看低信心":
            d2 = d2[d2.conf < 0.8]
        st.caption(f"{len(d2)} 張 · 顯示前 60 · 灰塊是遮掉的浮水印 · "
                   "紅色 TEST = 模型沒背過，只有這些算數")

        cols = st.columns(6)
        for i, r in enumerate(d2.head(60).itertuples()):
            src = paths.QMS_IMAGES / f"{r.fileId}.jpg"
            with cols[i % 6]:
                if src.exists():
                    st.image(str(src), width="stretch")
                conf = getattr(r, "conf", None)
                st.markdown(
                    f'<div class=pc>'
                    f'<div>{badge(str(getattr(r, "part", "train")))}'
                    f'<span class=m> {r.cls}</span></div>'
                    + verdict("系統判斷", getattr(r, "pred", None), r.cls)
                    + f'<div class=m>信心：{"－" if conf is None or conf != conf else f"{conf:.2f}"}</div>'
                    + f'<div class=m title="{txt(r.l3)}">項目：{txt(r.l3)}</div>'
                    + f'<div class=m>{txt(r.floor)} · {txt(r.room)}</div>'
                      f'</div>', unsafe_allow_html=True)

    if tabs[3]:
        runs = sorted([p for p in paths.REPORTS_OUT.glob("*qms*") if p.is_dir()], reverse=True) \
            if paths.REPORTS_OUT.exists() else []
        report_view(runs, "qms_run")
