"""階段 D：評估 + Gemini 基準線。

公平比較的唯一方式：把 Gemini 的 predWorkItem 用**完全相同**的 labels.yaml
規則正規化，再跟本地模型比同一份測試集。

    uv run --extra train src/evaluate.py [--split v1] [--baseline]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date

import numpy as np
import pandas as pd

from core import model_registry as registry
from core import paths
from labels import Labeler, labeled_manifest


def confusion_png(y_true, y_pred, labels, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from sklearn.metrics import confusion_matrix

    for f in ("PingFang TC", "Heiti TC", "Arial Unicode MS"):  # 中文標籤要有字型
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(labels), 1 + 0.6 * len(labels)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("預測")
    ax.set_ylabel("實際")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def gemini_baseline(
    df_test: pd.DataFrame,
    labeler: Labeler,
    valid: set[str],
    local_pred: dict[str, str] | None = None,
    detail_csv=None,
) -> dict:
    """predWorkItem（= photos[].anno.workItem）經同一套規則正規化後的準確率。

    valid = 本地模型實際存在的類別。Gemini 正規化後可能命中一個因
    min_class_size 被併掉的類（例：「電梯安裝工程」→ 設備-電梯停車），
    那是不存在的答案，必須跟真實標籤一樣落到 fallback，否則等於扣它冤枉分。
    """
    sub = df_test[df_test.predWorkItem.notna() & (df_test.predWorkItem.astype(str) != "")]
    if not len(sub):
        return {"support": 0, "note": "測試集內無 predWorkItem"}
    raw = sub.predWorkItem.map(lambda t: labeler.label(t) or labeler.fallback)
    pred = raw.map(lambda c: c if c in valid else labeler.fallback)
    local = sub.fileId.map(local_pred or {})
    if detail_csv is not None:
        sub.assign(
            geminiRaw=sub.predWorkItem,
            geminiNorm=pred,
            localPred=local,
            geminiHit=pred.values == sub.cls.values,
            localHit=local.values == sub.cls.values,
        )[["fileId", "title", "cls", "geminiRaw", "geminiNorm", "geminiHit", "localPred", "localHit"]].to_csv(
            detail_csv, index=False
        )
    out = {
        "support": len(sub),
        "coverage": round(len(sub) / max(len(df_test), 1), 3),
        "top1": round(float((pred.values == sub.cls.values).mean()), 3),
        "distinctRawStrings": int(sub.predWorkItem.nunique()),
        "note": "僅涵蓋有 anno.workItem 的子集，非全測試集",
    }
    if local_pred:  # 同一批照片上的本地成績，唯一公平的對照
        out["localTop1SameSubset"] = round(float((local.values == sub.cls.values).mean()), 3)
    return out


def coverage_curve(proba, y_true, classes) -> list[dict]:
    """信心門檻 → (自己處理的比例, 該批準確率)。混合架構的退場判準。

    「本地能不能取代 Gemini」不是比整體準確率，是問：在某個門檻下本地能自己吃掉
    多少比例、而錯誤率不高於 Gemini。剩下的才需要 fallback。

    註：門檻是拿測試集本身算的，會樂觀。要當上線依據得另切一份 validation。
    """
    conf = proba.max(1)
    pred = np.asarray(classes)[proba.argmax(1)]
    out = []
    for th in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf >= th
        if not m.sum():
            continue
        out.append(
            {
                "threshold": th,
                "coverage": round(float(m.mean()), 3),
                "accuracyOnCovered": round(float((pred[m] == y_true[m]).mean()), 3),
                "n": int(m.sum()),
            }
        )
    return out


def work_item_metrics(test_ids, proba, y_true, classes) -> dict:
    """同一張考卷改用工項融合（core.evaluation_metrics.fuse_work_items）再算一次。

    單張分數仍是主指標（journal 與跨版比較都看它）；這一欄回答「推論時若參考同工項
    另一張照片，能多對幾張」。v40 實測 +5pt，打底/粉光這類單張難分的受益最多。
    """
    from sklearn.metrics import f1_score

    from core.evaluation_metrics import fuse_work_items

    man = pd.read_csv(paths.MANIFEST, dtype=str, usecols=["fileId", "dailyReportInfoId", "title"])
    man = man.fillna("").set_index("fileId")
    keys = [
        f"{man.at[f, 'dailyReportInfoId']}|{man.at[f, 'title']}"
        if f in man.index and man.at[f, "dailyReportInfoId"]
        else ""
        for f in test_ids
    ]
    fused = fuse_work_items(proba, keys)
    pred = np.asarray(classes)[fused.argmax(1)]
    # macro 只平均考卷上真的有的類別，與主指標同一條規則（見 run() 的 ghosts 註解）
    macro = f1_score(y_true, pred, labels=sorted(set(y_true)), average="macro", zero_division=0)
    return {
        "top1": round(float((pred == y_true).mean()), 4),
        "macroF1": round(float(macro), 4),
        "coverageCurve": coverage_curve(fused, y_true, classes),
    }


def run(
    split_name: str = "v1",
    model_key: str | None = None,
    baseline: bool = True,
    run_tag: str = "probe",
    errors: int = 24,
    log=print,
) -> str:
    from sklearn.metrics import classification_report

    sp = registry.load_split(split_name)
    model_key = model_key or registry.encoder(split_name)
    cls = sp["labels"]
    ids, emb = registry.load_features(model_key)
    idx = {f: i for i, f in enumerate(ids)}

    test_ids = [f for f in sp["test"] if f in idx and f in cls]
    x = np.stack([emb[idx[f]] for f in test_ids])
    y = np.array([cls[f] for f in test_ids])

    clf = registry.load_probe(split_name, model_key)
    pred = clf.predict(x)

    outdir = paths.REPORTS_OUT / f"{date.today():%Y-%m-%d}-{run_tag}"
    (outdir / "errors").mkdir(parents=True, exist_ok=True)

    labels = sorted(set(y) | set(pred))
    rep = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    # macro 平均只算**測試集裡真的有的**類別。labels 含了模型誤猜、但一張真例都沒有的
    # 類別（support=0 → f1 恆 0），sklearn 的 macro avg 會把那個 0 算進去。
    # v19 就是這樣被一個 0 張的「防水-防水施作」從 0.856 拉到 0.778——
    # 那是分母裡多了一個幽靈，不是模型退步。
    ghosts = sorted(k for k in labels if rep[k]["support"] == 0)
    real = [rep[k]["f1-score"] for k in labels if rep[k]["support"] > 0]
    metrics = {
        "top1": round(float((pred == y).mean()), 4),
        "macroF1": round(float(np.mean(real)), 4),
        "macroF1Classes": len(real),
        "support": len(y),
        "perClass": {k: v for k, v in rep.items() if k in labels},
    }
    if ghosts:
        # 測不到不等於沒問題：模型還是會輸出這些類別，只是這份考卷驗不了它
        metrics["zeroSupportClasses"] = ghosts
        log(f"  ⚠ 測試集 0 張但模型猜得出來的類別：{ghosts}（未計入 macroF1）")
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x)
        metrics["coverageCurve"] = coverage_curve(proba, y, clf.classes_)
        metrics["workItemFusion"] = work_item_metrics(test_ids, proba, y, clf.classes_)
        log(
            f"  工項融合（同日報×同標題兄弟照）：top1={metrics['workItemFusion']['top1']}"
            f" macroF1={metrics['workItemFusion']['macroF1']}（單張 top1={metrics['top1']}）"
        )

    if baseline:
        # Gemini 只存在於日報那批；測試集若含 QMS 照片，這裡自動只取得到的部分
        man = labeled_manifest().set_index("fileId")
        rows = [f for f in test_ids if f in man.index]
        if rows:
            metrics["geminiBaseline"] = gemini_baseline(
                man.loc[rows].reset_index(),
                Labeler.load(),
                set(cls.values()),
                dict(zip(test_ids, pred)),
                outdir / "gemini_detail.csv",
            )

    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    confusion_png(y, pred, labels, outdir / "confusion.png")

    # 誤判樣本縮圖：肉眼看十張，通常比看十個指標更快找到問題
    wrong = [(f, t, p) for f, t, p in zip(test_ids, y, pred) if t != p][:errors]
    for f, t, p in wrong:
        src = paths.IMAGES / f"{f}.jpg"
        if src.exists():
            shutil.copy2(src, outdir / "errors" / f"真{t}_猜{p}_{f[:8]}.jpg")

    g = metrics.get("geminiBaseline", {})
    (outdir / "comparison.md").write_text(
        f"# 本地模型 vs Gemini（split={split_name}）\n\n"
        f"| 維度 | 本地線性探針 | Gemini |\n|---|---|---|\n"
        f"| top-1（全測試集 {metrics['support']} 張） | {metrics['top1']} | 無此資料 |\n"
        f"| top-1（同子集 {g.get('support', 0)} 張） | {g.get('localTop1SameSubset', 'n/a')} "
        f"| {g.get('top1', 'n/a')} |\n"
        f"| 覆蓋率 | 100% | {g.get('coverage', 0):.0%} |\n"
        f"| 延遲 | ~30ms（估） | 6~15s（實測） |\n"
        f"| 每千張成本 | 0 | 依 token 計價 |\n"
        f"| 新工種上線 | 需重訓 | 改一行 prompt |\n"
        f"| 離線可用 | 可 | 不可 |\n\n"
        f"註：{g.get('note', '')}；Gemini 原始字串 {g.get('distinctRawStrings', 0)} 種。\n"
    )

    (outdir / "config.json").write_text(
        json.dumps(
            {
                "labelsVersion": Labeler.load().version,
                "split": split_name,
                "encoder": model_key,
                "model": registry.probe_path(split_name, model_key).name,
            },
            ensure_ascii=False,
            indent=1,
        )
    )

    log(f"報告 → {outdir}  top1={metrics['top1']} macroF1={metrics['macroF1']}")
    return str(outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v1")
    ap.add_argument("--model", default=None, help="預設讀 split 檔的 encoder 欄")
    ap.add_argument("--baseline", action="store_true", default=True)
    ap.add_argument("--run", default="probe")
    a = ap.parse_args()
    run(a.split, a.model, a.baseline, a.run)
