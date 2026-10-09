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
    if (paths.SPLITS / f"{name}.json").exists() or any(paths.MODELS.glob(f"probe-*-{name}.pkl")):
        raise ValueError(f"{name} 已存在，請使用新的版本名稱。")
    if any(paths.REPORTS_OUT.glob(f"????-??-??-{name}")):
        raise ValueError(f"{name} 已有評估報告，請使用新的版本名稱。")


def next_version() -> str:
    """下一個沒用過的 vNN：取 split、探針、報告三處出現過的最大號 +1。

    只看 CURRENT 往上數會撞到「訓練到一半失敗、留下報告但沒有 split」的號碼；
    每日排程無人值守，版本名必須自己找得到。
    """
    seen = [p.stem for p in paths.SPLITS.glob("v*.json")]
    seen += [p.stem.rsplit("-", 1)[-1] for p in paths.MODELS.glob("probe-*-v*.pkl")]
    seen += [p.name[11:] for p in paths.REPORTS_OUT.glob("????-??-??-v*")]
    nums = [int(m.group(1)) for v in seen if (m := re.fullmatch(r"v(\d+)", v))]
    return f"v{max(nums, default=0) + 1}"


def build_split(name: str, log=print) -> dict:
    import numpy as np

    import features
    import split as split_mod

    sp = split_mod.build(name=name, log=log)
    encoder = sp.setdefault("encoder", features.DEFAULT_ENCODER)
    if not sp["train"] or not sp["test"]:
        raise ValueError("PMS 訓練或測試集為空，需要更多不同日期的有效照片。")
    if len({sp["labels"][fid] for fid in sp["train"]}) < 2:
        raise ValueError("PMS 訓練集至少需要兩個工種類別。")
    if set(sp.get("datasets", {}).values()) - {"pms", "crop"}:
        raise ValueError("PMS 專用切分出現其他資料源。")
    available = set()
    for prefix in ("", "crops-"):
        path = paths.FEATURES / f"{prefix}{encoder}.npz"
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


def training_steps(
    name: str, with_data: bool = False, with_explain: bool = True
) -> list[tuple[str, Callable]]:
    """建立步驟不執行工作；切換模型是另一個決定（操作台手動，或每日排程的 core.promotion 考卷）。

    with_explain=False 跳過圖像解釋：它最慢（約 12 秒／張）且不影響分類分數，
    操作台在快取缺席時會現算熱區，所以無人值守的每日重訓先跳過。
    """
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
        # 分類器用 DEFAULT_ENCODER；siglip 仍補抽——相似照片索引（孤兒院）與舊版模型還在用它
        *(
            (
                f"PMS 圖像特徵（{key}）",
                lambda log, key=key: features.extract(model_key=key, src="report", log=log),
            )
            for key in dict.fromkeys((features.DEFAULT_ENCODER, features.LEGACY_ENCODER))
        ),
        ("人工證據裁切特徵", lambda log: features.extract_crops(model_key=features.DEFAULT_ENCODER, log=log)),
        (f"PMS 分組切分 {name}", lambda log: build_split(name, log)),
        ("工種分類器訓練", lambda log: train.probe(split_name=name, log=log)),
        ("工種分類評估", lambda log: evaluate.run(split_name=name, run_tag=name, log=log)),
        *(
            [("更新圖像解釋", lambda log: explain.run_probe(split_name=name, log=log))]
            if with_explain
            else []
        ),
        ("更新學習紀錄", lambda log: log("學習紀錄 →", journal.write())),
    ]
    return steps


def run(name: str, with_data: bool = False, log=print, with_explain: bool = True) -> None:
    for title, fn in training_steps(name, with_data, with_explain):
        log(title)
        fn(log)
    log(f"{name} 已完成。請檢查評估報告，再明確切換使用版本。")
