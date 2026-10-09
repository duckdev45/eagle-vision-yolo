"""資料快取載入器：manifest / 本地預測。

全部走 Streamlit cache（以檔案 mtime 當 key），分頁模組只呼叫、不重寫——
同一份資料在三個分頁出現時只讀一次磁碟。
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import paths
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


@st.cache_data(show_spinner=False)
def local_preds(model_key: str | None = None, split_name: str = "") -> tuple[dict, set]:
    """{fileId: 本地模型預測}, {測試集 fileId}。缺模型或特徵就回空的。

    訓練集照片的預測是它自己背過的，偏樂觀，所以要標出來。
    """
    import pickle

    import numpy as np

    import split as split_mod

    model_key = model_key or split_mod.encoder(split_name or None)
    try:
        z = np.load(paths.FEATURES / f"{model_key}.npz", allow_pickle=True)
        with (paths.MODELS / f"probe-{model_key}-{split_name}.pkl").open("rb") as f:
            clf = pickle.load(f)["clf"]
        test = set(json.loads((paths.SPLITS / f"{split_name}.json").read_text())["test"])
    except (FileNotFoundError, KeyError):
        return {}, set()
    return dict(zip(z["fileIds"].tolist(), clf.predict(z["emb"]))), test
