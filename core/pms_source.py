"""PMS 施作項目照片的共用來源邊界。

sync 保留 WORK_ITEM 與 WORKFORCE 原始紀錄；工種分類只使用 WORK_ITEM。
出工職稱不是施作項目標籤，不能靠 title 或 tradeName 猜測來源。
"""

from __future__ import annotations

import pandas as pd

WORK_ITEM = "WORK_ITEM"


def work_items(df: pd.DataFrame) -> pd.DataFrame:
    """只保留來源明確的施作項目；缺來源時阻止分類，要求更新 manifest。"""
    if df.empty:
        return df.copy()
    if "source" not in df:
        raise ValueError("PMS manifest 缺 source，請重新同步日報以辨識施作項目與出工照。")
    keep = df.source.fillna("").astype(str).str.strip().eq(WORK_ITEM)
    if "dataset" in df:
        keep &= df.dataset.fillna("").astype(str).isin(["", "pms", "report"])
    return df.loc[keep].copy()
