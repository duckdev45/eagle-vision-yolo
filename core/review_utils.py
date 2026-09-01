# core/review_utils.py
"""複核佇列的分層計算（服務層實作）。

規則本來寫在 app.py 的 `review_queue()` 裡，跟 Streamlit 綁死 → 沒開瀏覽器就
看不到還積了多少。抽出來成純函式之後 `uv run src/review.py` 直接印、操作台
`ui/review_ui.py` 畫同一份，也測得動。

四種訊號，照「誰說的」分層——不是加總成一個分數。前三種問「標籤對不對」，
第四種問「模型會不會」，混成一個數字就看不出該先看哪一批。
"""

from __future__ import annotations

import os as _os
import sys as _sys

for _p in (
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json  # noqa: E402
import pickle  # noqa: E402

import pandas as pd  # noqa: E402

import paths  # noqa: E402
from core.labeler import Labeler  # noqa: E402

MARGIN_LOW = 0.25  # top1 與 top2 差距小於這個 = 模型在兩類之間猶豫，值得人看

TIER_NAMES = {
    4: "人寫的有異議",
    3: "模型＋Gemini 都不同意",
    2: "Gemini 有異議",
    1: "模型難分（低邊際・僅測試集）",
}


def scores(model_key: str = "siglip", split_name: str = "") -> tuple[dict, set]:
    """{fileId: (預測, 信心, 邊際)}, {測試集 fileId}。缺模型或特徵就回空的。

    用邊際不用信心：信心 0.9 但第二名 0.85 的照片，模型其實在兩類之間猶豫；
    信心 0.5 而第二名 0.05 的反而很篤定。邊際小 = 真的難分 = 值得人看。
    """
    import numpy as np

    try:
        z = np.load(paths.FEATURES / f"{model_key}.npz", allow_pickle=True)
        with (paths.MODELS / f"probe-{model_key}-{split_name}.pkl").open("rb") as f:
            clf = pickle.load(f)["clf"]
        test = set(json.loads((paths.SPLITS / f"{split_name}.json").read_text())["test"])
    except (FileNotFoundError, KeyError):
        return {}, set()
    p = clf.predict_proba(z["emb"])
    top = np.sort(p, axis=1)
    return (
        {
            f: (clf.classes_[i], float(top[n, -1]), float(top[n, -1] - top[n, -2]))
            for n, (f, i) in enumerate(zip(z["fileIds"].tolist(), p.argmax(1)))
        },
        test,
    )


def build(
    df: pd.DataFrame, lab: Labeler, sc: dict, test_ids: set, margin_low: float = MARGIN_LOW
) -> pd.DataFrame:
    """加上 tier / why / isTest 三欄，只留 tier > 0 的。純函式，好測。"""
    gem = df.predWorkItem.map(lambda t: (lab.label(t) or lab.fallback) if isinstance(t, str) and t else None)
    df = df.assign(
        gemNorm=gem,
        mPred=df.fileId.map(lambda f: (sc.get(f) or (None,))[0]),
        mConf=df.fileId.map(lambda f: (sc.get(f) or (None, None))[1]),
        mMargin=df.fileId.map(lambda f: (sc.get(f) or (None, None, None))[2]),
    )

    chips, spec = df.get("clsChips"), df.get("specTrade")
    F = pd.Series(False, index=df.index)
    # chips 落到 fallback 不算不一致——查驗重點寫的是「素地清理是否乾淨」這種驗收條件，
    # 本來就不含工種詞，套規則當然標不出來。那是「沒訊號」，不是「有異議」。
    # 不濾掉的話佇列會從 27 張暴增到 111 張，全是假警報。
    d_chips = chips.notna() & (chips != lab.fallback) & (chips != df.cls) if chips is not None else F
    d_spec = spec.notna() & (spec != df.cls.str.split("-").str[0]) if spec is not None else F
    d_gem = gem.notna() & (gem != df.cls)
    d_model = df.mPred.notna() & (df.mPred != df.cls)
    # 「模型難分」只在測試集上算數：訓練集那幾百張它背過，邊際再小也是假的猶豫。
    is_test = df.fileId.isin(test_ids)
    unsure = df.mMargin.notna() & (df.mMargin < margin_low) & is_test

    tier = pd.Series(0, index=df.index)
    tier[unsure] = 1  # 模型兩類難分
    tier[d_gem] = 2  # 只有機器有異議
    tier[d_model & d_gem] = 3  # 模型與 Gemini 都不同意 title
    tier[d_chips | d_spec] = 4  # 人寫的有異議 —— 最強
    why = [
        ", ".join(
            w
            for w, b in (("查驗項目", c), ("specKey", sp), ("Gemini", g), ("模型不同意", m), ("模型難分", u))
            if b
        )
        for c, sp, g, m, u in zip(d_chips, d_spec, d_gem, d_model, unsure)
    ]
    q = df.assign(tier=tier, why=why, isTest=is_test)
    return q[q.tier > 0]
