"""labels.yaml → 標籤。屬 derived 層，隨時可改、隨時重算。

規則引擎（Labeler / human_refs / review.csv 存取）的實體在 core/labeler.py
（服務層）；本檔保留 manifest 組裝（labeled_manifest / pending_classes /
unclaimed），它們讀 data/ 的檔案，屬資料編排層。
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/

import paths
from core.labeler import (
    SPEC_KEY_TRADE,
    Labeler,
    human_refs,
    load_boxes,
    load_reviews,
    orphan_reviews,
    save_review,
)
from core.pms_source import work_items


def pending_classes(df: pd.DataFrame = None, lab: Labeler = None) -> pd.DataFrame:
    """規則認得、但張數還沒到 `min_class_size` 的類別 → 每列一類。

    這些不是錯誤，是**在排隊**：張數一過門檻就自動進訓練，不用改任何程式
    （`金屬-欄杆鐵件` 就是這樣在 v18 自己冒出來的）。

    門檻算的是 **train + test 全部**，不是 train——`apply()` 在切分之前就先砍了。
    別跟 Makefile 的 `MIN_TRAIN` 搞混，那個是進場之後的切分保底。

    回傳欄位：cls / photos（現有）/ need（還差幾張）/ latest（最近一張的日期）。
    """
    lab = lab or Labeler.load()
    if df is None:
        from sync import _truthy

        df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
        if "active" in df:
            df = df[_truthy(df.active)]
        df = work_items(df)
    full = lab.apply(df, drop_small=False)
    full = full[full.cls != lab.fallback]
    n = full.cls.value_counts()
    small = n[n < lab.min_class_size]
    if not len(small):
        return pd.DataFrame(columns=["cls", "photos", "need", "latest"])
    latest = (
        full[full.cls.isin(small.index)].groupby("cls").reportDate.max()
        if "reportDate" in full
        else pd.Series(dtype=str)
    )
    return (
        pd.DataFrame(
            {
                "cls": small.index,
                "photos": small.values,
                "need": lab.min_class_size - small.values,
                "latest": [latest.get(c, "") for c in small.index],
            }
        )
        .sort_values("need")
        .reset_index(drop=True)
    )


def unclaimed(df: pd.DataFrame = None, lab: Labeler = None) -> pd.DataFrame:
    """規則沒認領、因此不會進訓練的照片（`cls == fallback`）。

    `apply()` 預設就把它們濾掉了，所以誰也看不到——操作台 ⑥ 那個
    「落入 fallback 的標題 —— 這裡就是下一條規則的來源」面板，實測永遠顯示 0 張，
    因為它拿到的 DataFrame 早就被濾過。這個函式繞開那層濾網。

    照片沒有被刪，都還在 raw/photos/。少的只是一條規則。
    """
    lab = lab or Labeler.load()
    if df is None:
        from sync import _truthy

        df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
        if "active" in df:
            df = df[_truthy(df.active)]
        df = work_items(df)
    out = lab.apply(df, drop_small=False)
    return out[out.cls == lab.fallback]


def legacy_manifest() -> pd.DataFrame:
    """舊 pptx 那批，欄位對齊到 PMS 的最小集合。

    `constrId` 用 constrName 對回 PMS——切分要按工地分組，名字對不上就分不了組。
    對不到的（PMS 沒有那個案場）用 `legacy:<名字>` 當自己的 id，同名的至少不會拆開。
    """
    if not paths.LEGACY_MANIFEST.exists():
        return pd.DataFrame()
    d = pd.read_csv(paths.LEGACY_MANIFEST, dtype=str)
    name2id = {}
    if paths.MANIFEST.exists():
        p = pd.read_csv(paths.MANIFEST, dtype=str, usecols=["constrName", "constrId"])
        name2id = dict(p.dropna().drop_duplicates().values)
    return pd.DataFrame(
        {
            "fileId": d.fileId,
            "dataset": "legacy",
            "title": d.title,
            "reportDate": d.reportDate.fillna(""),
            "constrName": d.constrName,
            "constrId": d.constrName.map(lambda n: name2id.get(n, f"legacy:{n}")),
        }
    )


def labeled_manifest(active_only: bool = True) -> pd.DataFrame:
    """PMS 日報施作照 → 加上 cls 欄（舊 pptx 不進訓練；G1 抽樣另用 legacy_manifest）。"""
    df = pd.read_csv(paths.MANIFEST)
    df = work_items(df)
    if active_only and "active" in df:
        df = df[df.active.astype(str).str.lower().isin(["true", "1"])]
    df = df.assign(dataset="pms")
    df = df.join(human_refs(df, Labeler.load()))
    return Labeler.load().apply(df)


if __name__ == "__main__":
    d = labeled_manifest()
    print(d.cls.value_counts().to_string())
    print(f"\n合計 {len(d)} 張 / {d.cls.nunique()} 類")
