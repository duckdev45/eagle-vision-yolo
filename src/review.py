"""複核佇列的分層規則。純函式，操作台與命令列共用同一份。

規則本來寫在 app.py 的 `review_queue()` 裡，跟 Streamlit 綁死 → 沒開瀏覽器就
看不到還積了多少。抽出來之後 `uv run src/review.py` 直接印，也測得動。

四種訊號，照「誰說的」分層——不是加總成一個分數。前三種問「標籤對不對」，
第四種問「模型會不會」，混成一個數字就看不出該先看哪一批。

    uv run src/review.py                # 還沒裁的，照優先序印
    uv run src/review.py --since 2026-08-19   # 某天之後才進來的
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paths  # noqa: E402
import split as split_mod  # noqa: E402
from labels import Labeler, human_refs, load_reviews  # noqa: E402

MARGIN_LOW = 0.25  # top1 與 top2 差距小於這個 = 模型在兩類之間猶豫，值得人看

TIER_NAMES = {4: "人寫的有異議", 3: "模型＋Gemini 都不同意", 2: "Gemini 有異議",
              1: "模型難分（低邊際・僅測試集）"}


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
    return ({f: (clf.classes_[i], float(top[n, -1]), float(top[n, -1] - top[n, -2]))
             for n, (f, i) in enumerate(zip(z["fileIds"].tolist(), p.argmax(1)))}, test)


def build(df: pd.DataFrame, lab: Labeler, sc: dict, test_ids: set,
          margin_low: float = MARGIN_LOW) -> pd.DataFrame:
    """加上 tier / why / isTest 三欄，只留 tier > 0 的。純函式，好測。"""
    gem = df.predWorkItem.map(
        lambda t: (lab.label(t) or lab.fallback) if isinstance(t, str) and t else None)
    df = df.assign(
        gemNorm=gem,
        mPred=df.fileId.map(lambda f: (sc.get(f) or (None,))[0]),
        mConf=df.fileId.map(lambda f: (sc.get(f) or (None, None))[1]),
        mMargin=df.fileId.map(lambda f: (sc.get(f) or (None, None, None))[2]))

    chips, spec = df.get("clsChips"), df.get("specTrade")
    F = pd.Series(False, index=df.index)
    # chips 落到 fallback 不算不一致——查驗重點寫的是「素地清理是否乾淨」這種驗收條件，
    # 本來就不含工種詞，套規則當然標不出來。那是「沒訊號」，不是「有異議」。
    # 不濾掉的話佇列會從 27 張暴增到 111 張，全是假警報。
    d_chips = (chips.notna() & (chips != lab.fallback) & (chips != df.cls)
               if chips is not None else F)
    d_spec = (spec.notna() & (spec != df.cls.str.split("-").str[0]) if spec is not None else F)
    d_gem = gem.notna() & (gem != df.cls)
    d_model = df.mPred.notna() & (df.mPred != df.cls)
    # 「模型難分」只在測試集上算數：訓練集那幾百張它背過，邊際再小也是假的猶豫。
    is_test = df.fileId.isin(test_ids)
    unsure = df.mMargin.notna() & (df.mMargin < margin_low) & is_test

    tier = pd.Series(0, index=df.index)
    tier[unsure] = 1                    # 模型兩類難分
    tier[d_gem] = 2                     # 只有機器有異議
    tier[d_model & d_gem] = 3           # 模型與 Gemini 都不同意 title
    tier[d_chips | d_spec] = 4          # 人寫的有異議 —— 最強
    why = [", ".join(w for w, b in (("查驗項目", c), ("specKey", sp), ("Gemini", g),
                                    ("模型不同意", m), ("模型難分", u)) if b)
           for c, sp, g, m, u in zip(d_chips, d_spec, d_gem, d_model, unsure)]
    q = df.assign(tier=tier, why=why, isTest=is_test)
    return q[q.tier > 0]


def _labeled() -> tuple[pd.DataFrame, Labeler]:
    """manifest + 規則標籤 + 人寫的兩個參考答案。與 app.labeled() 同一套，
    但不經過 Streamlit 的 cache_data（那個裝飾器在無 st context 下會炸）。"""
    from sync import _truthy
    df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
    if "active" in df:
        df = df[_truthy(df.active)]
    lab = Labeler.load()
    out = lab.apply(df)
    return out.join(human_refs(out, lab)), lab


def queue(split_name: str = "", since: str = "") -> pd.DataFrame:
    df, lab = _labeled()
    sc, test_ids = scores("siglip", split_name or split_mod.current())
    q = build(df, lab, sc, test_ids)
    done = set(load_reviews())
    q = q[~q.fileId.isin(done)]
    if since:
        q = q[q.reportDate.fillna("") >= since]
    return q.sort_values(["tier", "syncedAt", "reportDate"], ascending=False, kind="stable")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="", help="用哪一版模型算「難分」，預設 CURRENT")
    ap.add_argument("--since", default="", help="只看 reportDate >= 這天的")
    ap.add_argument("--limit", type=int, default=30)
    a = ap.parse_args()

    q = queue(a.split, a.since)
    cur = a.split or split_mod.current()
    print(f"模型 {cur} · 已裁 {len(load_reviews())} 筆 · 待裁 {len(q)} 張"
          + (f"（reportDate >= {a.since}）" if a.since else ""))
    if not len(q):
        print("佇列是空的。")
        sys.exit(0)
    for t in (4, 3, 2, 1):
        n = int((q.tier == t).sum())
        if n:
            print(f"  第 {t} 層 {TIER_NAMES[t]}：{n} 張")
    print(f"\n前 {min(a.limit, len(q))} 張（最強訊號在前）：")
    print(f"{'fileId':10} {'層':>2} {'reportDate':11} {'現在的標籤':14} {'模型猜':14} 訊號")
    for r in q.head(a.limit).itertuples():
        print(f"{str(r.fileId)[:8]:10} {r.tier:>2} {str(r.reportDate):11} "
              f"{str(r.cls):14} {str(r.mPred):14} {r.why}")
    print("\n裁決要人做：`make app` → ④ 複核佇列。裁完 `make model SPLIT=vN` 才會進模型。")
