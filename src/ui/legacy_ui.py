"""歷史資料頁（舊 pptx 進度報告）。"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import paths
import split as split_mod

from .common import txt
from .data import labeled


@st.cache_data(show_spinner=False)
def _legacy(_mtime: float) -> pd.DataFrame:
    from labels import Labeler, legacy_manifest

    lg = legacy_manifest()
    if not len(lg):
        return lg
    raw = pd.read_csv(paths.LEGACY_MANIFEST, dtype=str)
    lab = Labeler.load()
    return lg.assign(
        cls=lg.title.map(lab.label),
        dupCount=pd.to_numeric(raw.dupCount, errors="coerce").fillna(1).astype(int),
        pptx=raw.pptx,
        slide=raw.slide,
    )


def legacy_page() -> None:
    """歷史資料（舊 pptx 進度報告）—— 它補了什麼、有沒有進訓練集。"""
    st.subheader("歷史資料（舊版 pptx 進度報告）")
    if not paths.LEGACY_MANIFEST.exists():
        st.info(
            "還沒抽取。跑 `make legacy`（或 "
            "`uv run --extra train src/legacy.py --root ~/Downloads/115年日報表`）。"
        )
        return
    d = _legacy(paths.LEGACY_MANIFEST.stat().st_mtime)
    from labels import Labeler

    lab = Labeler.load()

    # 第一件要講清楚的事：這批到底有沒有在訓練
    cur = split_mod.current()
    sp = split_mod.load(cur) if (paths.SPLITS / f"{cur}.json").exists() else {}
    n_in = int(sp.get("trainLegacy") or 0)
    if n_in:
        st.success(
            f"線上的 **{cur}** 有收這批：train 裡有 {n_in} 張歷史照片"
            f"（{len(sp.get('classes') or [])} 類）。歷史照片**只進 train，不進 test**。"
        )
    else:
        st.warning(
            f"線上的 **{cur}** 沒有收這批（`source={sp.get('source', '?')}`）。"
            "要收：`make model SPLIT=vN LEGACY=1`，或 ① 同步那頁重跑時加上。"
        )
        st.caption(
            "預設不收是實測結果：只加那 11 個新類別就掉 22.6pt（0.815 → 0.589），"
            "代價來自類別空間變大——那些類別完全沒有 PMS 照片可以定界線。"
        )

    stat = {}
    sf = paths.LEGACY_MANIFEST.parent / "stats.json"
    if sf.exists():
        stat = json.loads(sf.read_text()).get("stat") or {}
    k = st.columns(5)
    k[0].metric("留下的照片", len(d))
    k[1].metric("pptx", stat.get("pptx", "—"))
    k[2].metric("工地", d.constrName.nunique())
    k[3].metric(
        "可訓練類別", int(d.cls.notna().sum() and d[d.cls.notna() & (d.cls != lab.fallback)].cls.nunique())
    )
    dates = d.reportDate.dropna()
    dates = dates[dates != ""]
    k[4].metric("日期範圍", f"{dates.min()[5:]} ~ {dates.max()[5:]}" if len(dates) else "—")

    if stat:
        st.markdown("**去重：抽出來一半是重複的**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "擋掉的": "同一份 pptx 裡重複引用",
                        "張數": stat.get("dupPhotos"),
                        "靠什麼": "sha1（位元組完全相同）",
                    },
                    {
                        "擋掉的": "不同 pptx 重壓縮過的同一張",
                        "張數": stat.get("dupVisual"),
                        "靠什麼": "dhash（sha1 抓不到）",
                    },
                    {
                        "擋掉的": "與 PMS 已下載的撞畫面",
                        "張數": stat.get("dupWithPms"),
                        "靠什麼": "dhash 比對 data/raw/photos",
                    },
                    {
                        "擋掉的": "太小（logo/圖示）",
                        "張數": stat.get("tinySkipped"),
                        "靠什麼": "< 200×200 px",
                    },
                    {
                        "擋掉的": "沒有工項標題的投影片",
                        "張數": stat.get("noTitleSlides"),
                        "靠什麼": "版面垂直位置認不到標題",
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"{stat.get('slides')} 張投影片 → {stat.get('workItemSlides')} 張是工項頁 "
            f"→ 留下 {len(d)} 張照片。pptx 一定會重新編碼貼進去的圖，"
            "所以 sha1 對跨檔重複無效。"
        )

    st.divider()
    st.markdown("**它補了哪些類別** —— 打勾的是 PMS 湊不到門檻、只有這批撐得起來的")
    pms = labeled()
    a = d[d.cls.notna() & (d.cls != lab.fallback)].cls.value_counts().rename("歷史")
    b = pms.cls.value_counts().rename("PMS") if len(pms) else pd.Series(dtype=int, name="PMS")
    cmp = pd.concat([a, b], axis=1).fillna(0).astype(int).sort_values("歷史", ascending=False)
    cmp["只有歷史資料有"] = cmp.PMS.eq(0)
    st.dataframe(cmp, width="stretch")
    drop = int((d.cls == lab.fallback).sum()) + int(d.cls.isna().sum())
    st.caption(
        f"落到 fallback 或被 junk 排除的 {drop} 張不進訓練"
        f"（`drop_fallback: true`）——那是一袋互不相干的東西，不是一個類別。"
        "PMS 欄是**可訓練張數**（已套 min_class_size），所以未達門檻的顯示 0。"
    )

    st.divider()
    c1, c2, c3 = st.columns([2, 2, 2])
    site = c1.selectbox("工地", ["全部", *sorted(d.constrName.dropna().unique().tolist())], key="lg_site")
    v = d if site == "全部" else d[d.constrName == site]
    kls = c2.selectbox("類別", ["全部", *sorted(v.cls.dropna().unique().tolist())], key="lg_cls")
    v = v if kls == "全部" else v[v.cls == kls]
    only_dup = c3.toggle(
        "只看被反覆引用的（dupCount > 1）",
        value=False,
        help="同一張出現在多份 pptx = 被當代表照反覆使用，通常畫面較好",
    )
    if only_dup:
        v = v[v.dupCount > 1]
    v = v.sort_values(["reportDate", "slide"], ascending=False, kind="stable")
    st.caption(f"{len(v)} 張 · 最新的前 60")
    cols = st.columns(6)
    for i, r in enumerate(v.head(60).itertuples()):
        img = paths.LEGACY_IMAGES / f"{r.fileId}.jpg"
        with cols[i % 6]:
            if img.exists():
                st.image(str(img), width="stretch")
            st.markdown(
                f"<div class=pc6><span class=m>{txt(r.cls)} · {txt(r.reportDate)}</span>"
                f'<div class=m title="{txt(r.title)}">{txt(r.title)}</div>'
                f"<div class=m>{txt(r.constrName)} · 引用 {r.dupCount} 次</div>"
                f'<div class=m title="{txt(r.pptx)}">{txt(r.pptx)} p.{txt(r.slide)}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
