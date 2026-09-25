"""Jev（TypeSafe System One）vs labels.yaml regex：同一份人工裁決上的 A/B。

問題很單純：**37 條 regex 與一顆語言決策模型，誰比較會把標題對到工種？**
而且 regex 丟掉的那一大桶（drop_fallback，455 種標題），Jev 救得回幾成？

答案不靠感覺，靠 data/review.csv——那是唯一照片層級的人工裁決（WORKFLOW.md S1）。
規則引擎沒有見過它（review.csv 是拿來蓋規則的，不是拿來訓規則的），
Jev 也沒有見過它，兩邊起跑線一樣。

三個 arm：
  regex       core.labeler.Labeler，現行線上邏輯
  jev         Jev，state 只有標題——與 regex **完全同樣的輸入**，這才是公平比較
  jev+refs    Jev，state 加 chipsOn / specKey（人寫的參考答案，regex 根本沒讀）
              贏了是加分，但不能記在「取代 regex」這筆帳上

指標與 g2_gate.py 同一套（coverage @ precision ≥ 0.90），這樣結果可以直接跟
reports/ 底下那 14 份實驗擺在一起比，不用另立一套分數。

    export TYPESAFE_API_KEY=...
    uv run src/jev_label_eval.py --dry-run          # 不打 API，先看送出去長什麼樣
    uv run src/jev_label_eval.py --limit 40         # 小樣本試水溫
    uv run src/jev_label_eval.py                    # 全量
    uv run src/jev_label_eval.py --fallback-probe   # regex 丟掉那桶，Jev 救得回多少

回應會快取在 data/derived/jev-cache.jsonl（key = state+questions 的 sha1），
重跑不重複付費，也讓結果可複現。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from math import comb
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jev_questions as Q
import paths
from core.evaluation_metrics import PRECISION_BAR, coverage_at_precision
from core.labeler import Labeler

# 與 src/qms.py、src/api.py 同慣例：金鑰寫 .env，不必每個 shell 手動 export。
load_dotenv()

CACHE = paths.DERIVED / "jev-cache.jsonl"
WORKERS = 8  # early access 限 1,200 req/min，8 條遠低於上限


# --- 資料 -------------------------------------------------------------------
def truth(split_name: str, limit: int | None) -> tuple[pd.DataFrame, list[str]]:
    """review.csv（人工裁決）⋈ manifest（標題）。只留 split 凍住的類別。"""
    classes = json.loads((paths.SPLITS / f"{split_name}.json").read_text())["classes"]
    rv = pd.read_csv(paths.REVIEW, dtype=str).drop_duplicates("fileId", keep="last")
    mf = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
    df = rv.merge(mf[["fileId", "title", "specKey", "chipsOn"]], on="fileId", how="inner")
    df = df[df.cls.isin(classes) & df.title.notna()].reset_index(drop=True)
    if limit:
        # 分層抽樣：小類別不能在試水溫階段整個消失，所以每類至少留 1 張。
        # 不用 groupby.apply——它會把 cls 抽成 index，欄位就掉了。
        frac = limit / len(df)
        keep = [i for _, g in df.groupby("cls") for i in g.index[: max(1, round(frac * len(g)))]]
        df = df.loc[keep]
    return df.reset_index(drop=True), classes


def fallback_pool(limit: int | None) -> pd.DataFrame:
    """regex 標不出來的照片：junk 排除掉的 + 落到 fallback 的。

    這桶就是 drop_fallback: true 整批丟掉的東西。
    問題不是「regex 標錯」，是「regex 沒有意見」——語意上未必真的沒訊號。

    **兩個來源都要吃**：日報 140 張 / 18 種標題，舊 pptx 745 張 / 201 種。
    只讀日報會漏掉 92% 的桶——legacy 正是 labels.yaml 註解裡說「455 種互不相干
    的標題」的主要出處，也是 drop_fallback 當初被打開的原因。
    """
    lab = Labeler.load()
    parts = []
    for src, path in (("daily", paths.MANIFEST), ("legacy", paths.LEGACY_MANIFEST)):
        if not path.exists():
            continue
        m = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        m = lab.drop_excluded(m)
        m = m.assign(regex=m.title.map(lab.label), src=src)
        parts.append(m[m.regex.isna() | (m.regex == lab.fallback)])
    pool = pd.concat(parts, ignore_index=True)
    # 按標題去重：輸入只有標題，同標題問一次就夠（同 A/B 那邊的標題層級紀律）。
    # 先記下每種標題的張數——決定要不要補規則時，看的是它能救回幾張照片。
    counts = pool.title.value_counts()
    pool = pool.drop_duplicates("title").assign(photos=lambda d: d.title.map(counts))
    pool = pool.sort_values("photos", ascending=False)
    return pool.head(limit) if limit else pool


# --- Jev --------------------------------------------------------------------
def _key(state: dict, classes: list[str], tag: str) -> str:
    blob = json.dumps([state, classes, tag, Q.MODEL], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def _cache() -> dict:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["key"]] = r["answer"]
    return out


def ask(df: pd.DataFrame, classes: list[str], *, with_refs: bool, dry: bool) -> list[dict]:
    tag = "refs" if with_refs else "title"
    states = [Q.state_of(r, with_human_refs=with_refs) for _, r in df.iterrows()]
    if dry:
        print(json.dumps({"state": states[0], "classes": classes}, ensure_ascii=False, indent=1))
        print(f"\n[dry-run] 會送出 {len(states)} 筆 / arm={tag}，未呼叫 API")
        return []

    from typesafe_sdk import ChoiceAnswer, NoulAnswer, TypeSafeClient

    cached = _cache()
    questions = Q.build(classes)
    client = TypeSafeClient()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    fh = CACHE.open("a")

    def one(state: dict) -> dict:
        k = _key(state, classes, tag)
        if k in cached:
            return cached[k]
        resp = client.system_one(state=state, questions=questions, model=Q.MODEL)
        trade, nc, oot = (resp.answers[k] for k in ("trade", "not_construction", "out_of_taxonomy"))
        # union 型別收斂：這三題的形狀是 jev_questions.build() 定死的，
        # 回來對不上就是契約被打破，該當場炸而不是往下帶著壞資料跑。
        assert isinstance(trade, ChoiceAnswer), f"trade 應為 Choice，收到 {type(trade).__name__}"
        assert isinstance(nc, NoulAnswer) and isinstance(oot, NoulAnswer), "兩題應為 Noul"
        ans = {
            "choice": trade.choice,
            "confidence": float(trade.confidence),
            "probabilities": {kk: float(v) for kk, v in trade.probabilities.items()},
            "not_construction": float(nc.noul),
            "out_of_taxonomy": float(oot.noul),
            "input_tokens": int(resp.usage.input_tokens or 0),
        }
        fh.write(json.dumps({"key": k, "answer": ans}, ensure_ascii=False) + "\n")
        fh.flush()
        return ans

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        out = list(pool.map(one, states))
    fh.close()
    return out


# --- 指標（coverage 與 g2_gate 共用 core.evaluation_metrics）-----------------
def mcnemar(a_ok: list[bool], b_ok: list[bool]) -> dict:
    """配對顯著性：兩個 arm 吃同一批樣本，看的是「誰翻誰」而不是兩個獨立比例。

    n 小又高度重疊時，比較 top-1 差幾 pt 沒有意義——能翻案的樣本才帶資訊。
    用精確二項檢定（不是卡方近似），b+c 只有個位數時近似會失準。
    """
    b = sum(x and not y for x, y in zip(a_ok, b_ok, strict=True))  # a 對 b 錯
    c = sum(y and not x for x, y in zip(a_ok, b_ok, strict=True))  # b 對 a 錯
    n = b + c
    p = sum(comb(n, k) for k in range(min(b, c) + 1)) * 2 / 2**n if n else 1.0
    return {"aOnly": b, "bOnly": c, "p": round(min(p, 1.0), 4)}


def score(df: pd.DataFrame, pred: list[str | None], conf: list[float]) -> dict:
    y = df.cls.tolist()
    pred = [None if pd.isna(p) else p for p in pred]
    correct = [p == t for p, t in zip(pred, y, strict=True)]
    abstained = [p is None for p in pred]
    coverage = coverage_at_precision(correct, conf, precision_bar=PRECISION_BAR, abstained=abstained)
    per_class: dict[str, dict] = {}
    for c in sorted(set(y)):
        idx = [i for i, t in enumerate(y) if t == c]
        per_class[c] = {
            "n": len(idx),
            "recall": round(sum(correct[i] for i in idx) / len(idx), 3),
            "本輪可評": "✅" if len(idx) >= 30 else f"n={len(idx)} <30",
        }
    return {
        "top1": round(sum(correct) / len(correct), 4) if correct else None,
        "coverage@p90": round(coverage, 4),
        "abstain": sum(abstained),
        "support": len(y),
        "perClass": per_class,
    }


def contradictions(df: pd.DataFrame) -> list[dict]:
    """同一個標題被人工標成多個類別＝ground truth 自己打架。

    這不是模型的問題，是 review.csv 的問題——但它會壓低**所有** arm 的天花板，
    不挑出來的話，兩邊都錯的那幾筆會被誤讀成「模型不行」。
    照片層級的裁決本來就可能與標題不同（一個工項的兩張照片常在拍不同階段），
    所以這裡只是回報、不是判錯，要人看圖才知道是誰錯。
    """
    out = []
    for t, g in df.groupby("title"):
        if g.cls.nunique() > 1:
            out.append({"title": t, "labels": sorted(g.cls.unique()), "n": len(g)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="預設讀 splits/CURRENT")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fallback-probe", action="store_true", help="改測 regex 丟掉那桶")
    ap.add_argument("--tag", default="jev")
    a = ap.parse_args()

    if not a.dry_run and not os.getenv("TYPESAFE_API_KEY"):
        print("需要 TYPESAFE_API_KEY（console.typesafe.ai/settings/keys）", file=sys.stderr)
        return 2

    split_name = a.split or (paths.SPLITS / "CURRENT").read_text().strip()
    classes = json.loads((paths.SPLITS / f"{split_name}.json").read_text())["classes"]
    outdir = paths.ROOT / "reports" / f"{date.today().isoformat()}-{a.tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 模式二：regex 沒有意見的那桶，Jev 救得回多少 ----
    if a.fallback_probe:
        pool = fallback_pool(a.limit)
        print(f"regex 無意見的標題：{len(pool)} 種")
        ans = ask(pool, classes, with_refs=False, dry=a.dry_run)
        if not ans:
            return 0
        pool = pool.assign(
            jev=[x["choice"] for x in ans],
            conf=[x["confidence"] for x in ans],
            notConstruction=[x["not_construction"] for x in ans],
            outOfTaxonomy=[x["out_of_taxonomy"] for x in ans],
        )
        # 「救回來」的定義寫死在這裡，不靠讀報告的人自己心算：
        # 模型選了真類別（非其他）、信心過自動採用線、且不認為它在分類樹外。
        rescued = pool[
            (pool.jev != Q.OTHER) & (pool.conf >= Q.AUTO_ACCEPT_CONF) & (pool.outOfTaxonomy < Q.OOD_NOUL)
        ]
        res = {
            "mode": "fallback-probe",
            "split": split_name,
            "titles": len(pool),
            "photos": int(pool.photos.sum()),
            "rescuedTitles": len(rescued),
            "rescuedPhotos": int(rescued.photos.sum()),
            "rescuedTitlePct": round(len(rescued) / len(pool), 4),
            "rescuedPhotoPct": round(rescued.photos.sum() / pool.photos.sum(), 4),
            "bySource": pool.src.value_counts().to_dict(),
            "byClass": rescued.groupby("jev").photos.sum().sort_values(ascending=False).to_dict(),
            "thresholds": {"conf": Q.AUTO_ACCEPT_CONF, "ood": Q.OOD_NOUL},
        }
        (outdir / "fallback_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        cols = ["title", "photos", "src", "jev", "conf", "outOfTaxonomy", "notConstruction"]
        pool[cols].to_csv(outdir / "fallback_probe.csv", index=False)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        # 補規則的決策看「能救回幾張照片」，不是幾種標題——一種寫法可能只有 1 張。
        print("\n=== 救回最多照片的前 25 種標題 ===")
        print(rescued.head(25)[["title", "photos", "src", "jev", "conf"]].to_string(index=False))
        print(f"\n逐筆 → {outdir}/fallback_probe.csv（要人看圖抽驗才算數）")
        return 0

    # ---- 模式一：三個 arm 打同一份人工裁決 ----
    df, classes = truth(split_name, a.limit)
    print(
        f"人工裁決樣本 {len(df)} 張 / {df.cls.nunique()} 類 / "
        f"**{df.title.nunique()} 種標題** / split={split_name}"
    )

    lab = Labeler.load()
    rx = [lab.label(t) for t in df.title]
    # regex 沒有信心值可言：命中就是 1.0，沒意見就是 0.0。
    # 命中者同分，只能整組接受或排除；沒意見者棄權，不計入接受數。
    rx_pred = [None if p in (None, lab.fallback) else p for p in rx]
    arms = {"regex": score(df, rx_pred, [0.0 if p is None else 1.0 for p in rx_pred])}

    rows = df.assign(regex=pd.Series(rx_pred, dtype="object"))
    for name, refs in (("jev", False), ("jev+refs", True)):
        ans = ask(df, classes, with_refs=refs, dry=a.dry_run)
        if not ans:
            return 0
        pred = [None if x["choice"] == Q.OTHER else x["choice"] for x in ans]
        arms[name] = score(df, pred, [x["confidence"] for x in ans])
        arms[name]["inputTokens"] = sum(x["input_tokens"] for x in ans)
        rows[name] = pred
        rows[f"{name}_conf"] = [x["confidence"] for x in ans]

    # 照片層級會嚴重灌水：輸入只有標題，同標題的 N 張照片是**同一個判斷**被計分 N 次
    # （實測 402 張只有 100 種標題，最兇的一個標題佔 58 張）。
    # 這與 WORKFLOW.md §2 鐵律 3「畫面幾乎相同的不可跨 train/test」同一個道理：
    # 重複樣本不帶新資訊，卻會讓差距看起來有統計效力。**判斷一律看標題層級。**
    uniq = rows.drop_duplicates("title")
    by_title = {}
    for name in arms:
        p = uniq[name].tolist()
        c = [0.0 if x is None else 1.0 for x in p] if name == "regex" else uniq[f"{name}_conf"].tolist()
        by_title[name] = score(uniq, p, c)
    ok = {n: (uniq.cls == uniq[n]).tolist() for n in arms}
    paired = {
        "regex_vs_jev": mcnemar(ok["regex"], ok["jev"]),
        "regex_vs_jev+refs": mcnemar(ok["regex"], ok["jev+refs"]),
    }

    metrics = {
        "mode": "ab",
        "split": split_name,
        "labelsVersion": lab.version,
        "model": Q.MODEL,
        "truth": "data/review.csv（人工裁決）",
        "precisionBar": PRECISION_BAR,
        "photos": len(df),
        "titles": int(df.title.nunique()),
        "byTitle": by_title,  # ← 下判斷看這個
        "byPhoto": arms,  # ← 參考用，同標題重複計分
        "paired": paired,
        "groundTruthConflicts": contradictions(df),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    rows.to_csv(outdir / "per_photo.csv", index=False)

    hdr = "| arm | top-1 | coverage@p90 | 棄權 |\n|---|---|---|---|\n"
    tbl = "".join(
        f"| {k} | {v['top1']} | {v['coverage@p90']} | {v['abstain']}/{v['support']} |\n"
        for k, v in by_title.items()
    )
    pair = "".join(f"| {k} | {v['aOnly']} | {v['bOnly']} | {v['p']} |\n" for k, v in paired.items())
    conf_md = "".join(
        f"- `{c['title']}`（{c['n']} 張）→ {' / '.join(c['labels'])}\n"
        for c in metrics["groundTruthConflicts"]
    )
    (outdir / "REPORT.md").write_text(
        f"# Jev vs regex（{split_name}，labels.yaml v{lab.version}，{Q.MODEL}）\n\n"
        f"## 標題層級（下判斷看這張）\n\n{hdr}{tbl}\n"
        f"樣本：{len(uniq)} 種標題（來自 {len(df)} 張人工裁決照片）。\n"
        f"輸入只有標題，所以同標題的多張照片是**同一個判斷**，照片層級計分會灌水"
        f"（最兇的一個標題佔 58 張）。照片層級數字留在 `metrics.json` 的 `byPhoto`。\n\n"
        f"## 配對檢定（McNemar 精確二項）\n\n"
        f"| 對比 | 前者贏 | 後者贏 | p |\n|---|---|---|---|\n{pair}\n"
        f"只有「一方對、另一方錯」的樣本帶資訊；兩邊都對或都錯的不影響結論。\n\n"
        f"## 真標籤自相矛盾（{len(metrics['groundTruthConflicts'])} 組）\n\n"
        f"{conf_md or '（無）'}\n"
        f"同一標題被標成不同類別。照片層級裁決本來就可能與標題不同"
        f"（一個工項的兩張照片常在拍不同階段），所以這是**待人看圖確認**的清單，"
        f"不是自動判錯。它會同時壓低所有 arm 的天花板。\n\n"
        f"## arm 定義\n\n"
        f"`jev` 與 `regex` 吃**同樣的輸入**（只有標題）；`jev+refs` 另外吃了 "
        f"chipsOn/specKey，regex 讀不到那兩欄，**不列入取代與否的判斷**。\n"
    )
    print(hdr + tbl)
    print(f"配對檢定：{json.dumps(paired, ensure_ascii=False)}")
    print(f"真標籤矛盾 {len(metrics['groundTruthConflicts'])} 組")
    print(f"報告 → {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
