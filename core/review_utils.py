# core/review_utils.py
"""複核佇列的分層計算（服務層實作）。

規則本來寫在 app.py 的 `review_queue()` 裡，跟 Streamlit 綁死 → 沒開瀏覽器就
看不到還積了多少。抽出來成純函式之後 `uv run src/review.py` 直接印、操作台
`ui/review_ui.py` 畫同一份，也測得動。

四種訊號，照「誰說的」分層——不是加總成一個分數。前三種問「標籤對不對」，
第四種問「模型會不會」，混成一個數字就看不出該先看哪一批。
"""

from __future__ import annotations

import pandas as pd

from core import model_registry as registry
from core.labeler import Labeler

MARGIN_LOW = 0.25  # top1 與 top2 差距小於這個 = 模型在兩類之間猶豫，值得人看

TIER_NAMES = {
    4: "人寫的有異議",
    3: "模型＋Gemini 都不同意",
    2: "Gemini 有異議",
    1: "模型難分（低邊際・僅測試集）",
}


def scores(model_key: str | None = None, split_name: str = "") -> tuple[dict, set]:
    """{fileId: (預測, 信心, 邊際)}, {測試集 fileId}。缺模型或特徵就回空的。

    用邊際不用信心：信心 0.9 但第二名 0.85 的照片，模型其實在兩類之間猶豫；
    信心 0.5 而第二名 0.05 的反而很篤定。邊際小 = 真的難分 = 值得人看。
    """
    import numpy as np

    model_key = model_key or registry.encoder(split_name or None)
    try:
        z = np.load(registry.feature_path(model_key), allow_pickle=True)
        clf = registry.load_probe(split_name, model_key)
        test = set(registry.load_split(split_name)["test"])
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


def orphans(df: pd.DataFrame, lab: Labeler) -> pd.DataFrame:
    """異類隔離區：規則沒命中的照片（cls = NaN 或 fallback）。

    `Labeler.apply` 的 `drop_fallback` 把這批排拒在訓練與複核佇列之外——那對
    訓練是對的（「其他 = 一袋雜物」實測 top-1 0.760 → 0.699），但「不倒進其他」
    不等於「該丟掉」。人看一張裁一張，裁決寫進 review.csv 之後 `apply` 會收回來。

    回傳加 `orphanWhy` 欄：`fallback`（有字但沒規則接）或 `無規則命中`（title 是
    樓層編號之類的無語意字串）。純函式，操作台與 CLI 共用。
    """
    out = lab.apply(df, drop_small=False)
    m = out.cls.isna() | (out.cls == lab.fallback)
    o = out[m].copy()
    o["orphanWhy"] = "fallback"
    o.loc[o.cls.isna(), "orphanWhy"] = "無規則命中"
    return o


def embedding_index(model_key: str = "siglip", _mtime: float = 0.0) -> tuple[list[str], object]:
    """L2 正規化後的 embedding 矩陣（fileIds, ndarray n×d），全資料源合併。

    走 `model_registry.load_features`（主檔 + legacy + 裁切合流）——孤兒與鄰居
    兩邊都可能來自任何源，只讀主檔的話 legacy 孤兒會查無此人。mtime 進 cache
    key 的責任在呼叫端（Streamlit cache_data 以參數為 key）。
    """
    import numpy as np

    ids, emb = registry.load_features(model_key)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return ids, emb / norm


def neighbors(
    file_id: str,
    labels: dict,
    model_key: str = "siglip",
    k: int = 5,
    _index: tuple[list[str], object] | None = None,
) -> list[tuple[str, str, float]]:
    """某張照片在 SigLIP 空間的 k 個最近**已分類**鄰居：[(fileId, cls, cosine)]。

    孤兒照片裁決的依據不是模型預測（它只會在既有類別裡挑一個），是「隔壁那五張
    長什麼樣、人給它們標了什麼」。cosine 相似度，排除自己；標籤來自 split 的
    labels dict（train+test 都算——孤兒要的是參考答案，不是考題）。
    """
    import numpy as np

    ids, emb = _index if _index is not None else embedding_index(model_key)
    try:
        i = ids.index(file_id)
    except ValueError:
        return []
    sims = emb @ emb[i]
    out: list[tuple[str, str, float]] = []
    for j in np.argsort(-sims):
        f = ids[int(j)]
        if f == file_id:
            continue
        c = labels.get(f)
        if c is None:
            continue
        out.append((f, c, float(sims[j])))
        if len(out) >= k:
            break
    return out
