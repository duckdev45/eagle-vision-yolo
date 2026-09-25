"""PMS 工種訓練的共同入口；CLI 與操作台消費同一份步驟。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import paths
from core import pms_review


def check_new_run(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("模型版本請使用英數字、連字號或底線，最多 64 字。")
    if (paths.SPLITS / f"{name}.json").exists() or (paths.MODELS / f"probe-siglip-{name}.pkl").exists():
        raise ValueError(f"{name} 已存在，請使用新的版本名稱。")
    if any(paths.REPORTS_OUT.glob(f"????-??-??-{name}")):
        raise ValueError(f"{name} 已有評估報告，請使用新的版本名稱。")


def build_split(name: str, log=print) -> dict:
    import numpy as np

    import split as split_mod

    sp = split_mod.build(name=name, with_legacy=False, legacy_fill=0, log=log)
    if not sp["train"] or not sp["test"]:
        raise ValueError("PMS 訓練或測試集為空，需要更多不同日期的有效照片。")
    if len({sp["labels"][fid] for fid in sp["train"]}) < 2:
        raise ValueError("PMS 訓練集至少需要兩個工種類別。")
    if set(sp.get("datasets", {}).values()) - {"pms", "crop"}:
        raise ValueError("PMS 專用切分出現其他資料源。")
    available = set()
    for prefix in ("", "crops-"):
        path = paths.FEATURES / f"{prefix}siglip.npz"
        if path.exists():
            with np.load(path, allow_pickle=True) as z:
                available.update(z["fileIds"].tolist())
    missing = set(sp["train"] + sp["test"]) - available
    if missing:
        raise ValueError(f"{len(missing)} 張照片缺少圖像特徵，請檢查前處理結果：{sorted(missing)[:5]}")
    sp["pmsCatalogVersion"] = pms_review.catalog_version()
    sp["pmsCatalog"] = pms_review.catalog()
    sp["trainingPolicy"] = "pms-only; human overrides title rules; pending candidates excluded"
    (paths.SPLITS / f"{name}.json").write_text(json.dumps(sp, ensure_ascii=False, indent=1), encoding="utf-8")
    return sp


def training_steps(name: str, with_data: bool = False) -> list[tuple[str, Callable]]:
    """建立步驟不執行工作；切換模型仍需獨立操作。"""
    check_new_run(name)
    import evaluate
    import explain
    import features
    import journal
    import prepare
    import sync
    import train

    def sync_data(log):
        result = sync.sync(log=log)
        if result.get("failed", 0):
            raise RuntimeError("部分 PMS 照片同步失敗，請檢查檔案。")

    def prepare_data(log):
        result = prepare.run(kind="report", log=log)
        if result.get("failed", 0):
            raise RuntimeError("部分 PMS 照片無法前處理，請檢查檔案。")

    steps = [("同步 PMS 日報", sync_data)] if with_data else []
    steps += [
        ("PMS 照片前處理", prepare_data),
        ("PMS 圖像特徵", lambda log: features.extract(src="report", log=log)),
        ("人工證據裁切特徵", lambda log: features.extract_crops(log=log)),
        (f"PMS 分組切分 {name}", lambda log: build_split(name, log)),
        ("工種分類器訓練", lambda log: train.probe(split_name=name, log=log)),
        ("工種分類評估", lambda log: evaluate.run(split_name=name, run_tag=name, log=log)),
        ("更新圖像解釋", lambda log: explain.run_probe(split_name=name, log=log)),
        ("更新學習紀錄", lambda log: log("學習紀錄 →", journal.write())),
    ]
    return steps


def run(name: str, with_data: bool = False, log=print) -> None:
    for title, fn in training_steps(name, with_data):
        log(title)
        fn(log)
    log(f"{name} 已完成。請檢查評估報告，再明確切換使用版本。")
