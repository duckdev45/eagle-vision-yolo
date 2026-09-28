"""標籤稽核：把測試集判錯的照片分成「標籤有問題」與「模型有問題」。

同一張照片有三個互相獨立的判斷：

    真值 cls    人工日報 title 經 labels.yaml 推出來的
    Gemini      anno.workItem 經**同一份** labels.yaml 正規化（與 evaluate.py 同一套）
    local       線性探針的預測

兩兩比較就能把錯誤分桶。local 與 Gemini 都指向同一個「別的東西」時，
可疑的是標籤而不是模型——兩個獨立系統不會犯一模一樣的錯。

⚠ 這支程式會讀測試集。產出只能拿來判斷「**哪一類**的標籤定義有問題」，
   不可以拿去逐張改答案：那等於把測試集的答案抄進訓練資料，
   之後所有數字都不算數。要動就動 labels.yaml 的規則或整類去留，
   改完重切分、重訓，並在報告裡記下這是第幾次依測試集資訊調整。

    uv run --extra train src/audit.py [--split v1] [--model siglip]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import boxes
import paths
import split as split_mod
from labels import Labeler

# 佐證框大到這個比例以上，等於「整張圖都是證據」——沒有定位，證據力當作零
BIG_BOX = 0.80
MANUAL_REVIEW = 10  # 排給人工目視的張數（框有沒有框到人，沒有自動判法）

CODES = {
    "A": "標籤可疑：local 與 Gemini 都判成同一個別的類別",
    "B": "多工項：真值出現在 Gemini 的次要工項裡，照片同時有兩件事",
    "C": "模型錯：Gemini 判對了，只有 local 錯",
    "R": "規則漏接：Gemini 講了具體工項，但 labels.yaml 沒有對應規則（落到 fallback）",
    "D": "不明：三者互不相同，且 Gemini 明確判成另一個現有類別",
    "E": "無法歸類：這張沒有 Gemini 輸出",
}


def _norm(labeler: Labeler, valid: set[str], text) -> str | None:
    """Gemini 的自由字串 → 我們的類別。

    正規化後可能命中一個因 min_class_size 被丟掉的類，那是不存在的答案，
    必須跟真實標籤一樣落到 fallback，否則等於扣它冤枉分（同 evaluate.py）。
    """
    if text is None or text != text or not str(text).strip():
        return None
    c = labeler.label(str(text)) or labeler.fallback
    return c if c in valid else labeler.fallback


def _secondary(labeler: Labeler, valid: set[str], raw) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out = [_norm(labeler, valid, t) for t in items if isinstance(t, str)]
    return [c for c in out if c]


def _missing(v) -> bool:
    """None 與 pandas 的 NaN 都算沒有。NaN != NaN 是唯一可靠的判法。"""
    return v is None or v != v or str(v).strip() == ""


def code(truth: str, gem, sec: list[str], loc: str, fallback: str = "其他") -> str:
    """錯誤歸類。順序即優先權。

    R 要排在 D 前面：Gemini 正規化落到 fallback，代表它講了一個具體工項而
    labels.yaml 沒有對應規則。那是規則覆蓋率的問題，跟「三方各說各話」不是同一件事，
    混在一起會讓人以為照片很難判，其實只是規則漏了一個詞。
    """
    if _missing(gem):
        return "E"
    if gem == truth:
        return "C"
    if loc == gem:
        return "A"
    if truth in sec:
        return "B"
    if gem == fallback:
        return "R"
    return "D"


def build(split_name: str = "v1", model_key: str | None = None, log=print) -> pd.DataFrame:
    import numpy as np

    import features

    sp = split_mod.load(split_name)
    model_key = model_key or split_mod.encoder(split_name)
    cls = sp["labels"]
    valid = set(sp.get("classes") or cls.values())
    labeler = Labeler.load()

    ids, emb = features.load(model_key)
    idx = {f: i for i, f in enumerate(ids)}
    test_ids = [f for f in sp["test"] if f in idx and f in cls]
    x = np.stack([emb[idx[f]] for f in test_ids])

    with (paths.MODELS / f"probe-{model_key}-{split_name}.pkl").open("rb") as fh:
        clf = pickle.load(fh)["clf"]
    pred = clf.predict(x)
    proba = clf.predict_proba(x).max(1) if hasattr(clf, "predict_proba") else [None] * len(x)

    man = pd.read_csv(paths.MANIFEST).drop_duplicates("fileId").set_index("fileId")
    rows = []
    for f, p, pr in zip(test_ids, pred, proba):
        m = man.loc[f] if f in man.index else None
        gem = _norm(labeler, valid, m.predWorkItem if m is not None else None)
        sec = _secondary(labeler, valid, m.predSecondary if m is not None else None)
        rows.append(
            {
                "fileId": f,
                "title": m.title if m is not None else None,
                "truth": cls[f],
                "gemini": gem,
                "geminiRaw": m.predWorkItem if m is not None else None,
                "geminiConf": m.predConf if m is not None else None,
                "secondary": "|".join(sec),
                "local": p,
                "localProba": round(float(pr), 3) if pr is not None else None,
                "maxBoxArea": round(boxes.area(m.predBoxes if m is not None else None), 3),
                "promptVersion": m.promptVersion if m is not None else None,
                "hit": p == cls[f],
            }
        )
    df = pd.DataFrame(rows)
    df["code"] = [
        ""
        if r.hit
        else code(r.truth, r.gemini, r.secondary.split("|") if r.secondary else [], r.local, labeler.fallback)
        for r in df.itertuples()
    ]
    log(f"測試集 {len(df)} 張，判錯 {int((~df.hit).sum())} 張")
    return df


def summarize(df: pd.DataFrame) -> str:
    wrong = df[~df.hit]
    out = [f"# 標籤稽核（測試集 {len(df)} 張，判錯 {len(wrong)} 張）", ""]
    out += ["> 產出只用來判斷**哪一類**的標籤定義有問題。逐張改答案 = 污染測試集。", ""]

    out += ["## 錯誤分桶", "", "| 代碼 | 意思 | 張數 | 佔錯誤 |", "|---|---|---|---|"]
    n = len(wrong) or 1
    for c, desc in CODES.items():
        k = int((wrong.code == c).sum())
        out.append(f"| {c} | {desc} | {k} | {k / n:.0%} |")

    out += [
        "",
        "## 各類別（依錯誤數排序）",
        "",
        "| 真值類別 | 測試張數 | 判錯 | " + " | ".join(CODES) + " |",
        "|---|---|---|" + "---|" * len(CODES),
    ]
    for c, g in sorted(df.groupby("truth"), key=lambda kv: -int((~kv[1].hit).sum())):
        w = g[~g.hit]
        cnt = {k: int((w.code == k).sum()) for k in CODES}
        out.append(f"| {c} | {len(g)} | {len(w)} | " + " | ".join(str(cnt[k]) for k in CODES) + " |")

    gap = wrong[wrong.code == "R"]
    if len(gap):
        out += [
            "",
            "## 規則漏接的 Gemini 原文（R）",
            "",
            "Gemini 講了具體工項，`labels.yaml` 沒有對應規則所以落到 fallback。"
            "這一段是規則要補什麼的直接清單。",
            "",
            "| Gemini 原文 | 現在的真值 | 張數 |",
            "|---|---|---|",
        ]
        for (raw, t), g in sorted(gap.groupby(["geminiRaw", "truth"]), key=lambda kv: -len(kv[1])):
            out.append(f"| {raw} | {t} | {len(g)} |")

    out += ["", "## 錯誤流向（真值 → local 判成什麼）", ""]
    for (t, p), g in sorted(wrong.groupby(["truth", "local"]), key=lambda kv: -len(kv[1]))[:12]:
        out.append(f"- {t} → **{p}** ×{len(g)}（{'/'.join(sorted(set(g.code)))}）")

    has = df[df.maxBoxArea > 0]
    out += ["", f"## 佐證框（{len(has)} 張有框）", ""]
    if len(has):
        big = int((has.maxBoxArea >= BIG_BOX).sum())
        out += [
            f"- 最大框面積中位數 **{has.maxBoxArea.median():.0%}**",
            f"- 框住 ≥{BIG_BOX:.0%} 畫面的有 **{big} 張（{big / len(has):.0%}）**"
            "——那種框等於沒定位，它的 workItem 證據力要打折",
            f"- `manual_review/` 放了面積最大的 {MANUAL_REVIEW} 張（已畫框）。"
            "「有沒有框到人」沒有自動判法，要人眼看。",
        ]
    else:
        out.append("- 測試集內沒有任何佐證框")
    return "\n".join(out) + "\n"


def run(split_name: str = "v1", model_key: str | None = None, log=print) -> str:
    df = build(split_name, model_key, log)
    outdir = paths.REPORTS_OUT / f"{date.today():%Y-%m-%d}-label-audit"
    review = outdir / "manual_review"
    review.mkdir(parents=True, exist_ok=True)

    df[~df.hit].sort_values(["truth", "code"]).to_csv(outdir / "triage.csv", index=False)
    df.to_csv(outdir / "all.csv", index=False)
    (outdir / "summary.md").write_text(summarize(df))

    # 人工目視：框最大的幾張，畫上框再另存（原圖不動）
    man = pd.read_csv(paths.MANIFEST).drop_duplicates("fileId").set_index("fileId")
    top = df[df.maxBoxArea > 0].nlargest(MANUAL_REVIEW, "maxBoxArea")
    for r in top.itertuples():
        src = paths.IMAGES / f"{r.fileId}.jpg"
        if src.exists():
            boxes.draw(str(src), man.predBoxes.get(r.fileId)).save(
                review / f"{r.maxBoxArea:.0%}_{r.truth}_{r.fileId[:8]}.jpg", "JPEG", quality=90
            )

    log(f"稽核 → {outdir}")
    return str(outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v1")
    ap.add_argument("--model", default=None, help="預設讀 split 檔的 encoder 欄")
    a = ap.parse_args()
    run(a.split, a.model)
