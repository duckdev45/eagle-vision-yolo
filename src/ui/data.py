"""資料快取載入器：manifest。

全部走 Streamlit cache（以檔案 mtime 當 key），分頁模組只呼叫、不重寫——
同一份資料在三個分頁出現時只讀一次磁碟。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core import paths
from labels import Labeler


@st.cache_data(show_spinner=False)
def _manifest(_mtime: float) -> pd.DataFrame:
    if not paths.MANIFEST.exists():
        return pd.DataFrame()
    df = pd.read_csv(paths.MANIFEST)
    return df[df.active.astype(str).str.lower().isin(["true", "1"])] if "active" in df else df


def load_manifest() -> pd.DataFrame:
    return _manifest(paths.MANIFEST.stat().st_mtime if paths.MANIFEST.exists() else 0.0)


def labeled() -> pd.DataFrame:
    """加上 cls，並帶上人寫的兩個參考答案（clsChips / specTrade）給複核佇列用。"""
    from core.pms_source import work_items
    from labels import human_refs

    df = work_items(load_manifest())
    if not len(df):
        return df
    lab = Labeler.load()
    out = lab.apply(df)
    return out.join(human_refs(out, lab))
