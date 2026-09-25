"""G2 promotion gate（WORKFLOW §7.2）：候選模型上線前的機械化檢查。

「什麼分數才能上線」有定義，就不必每次靠開會決定。本腳本跑 §7.2 裡
能離線算的三項（洩漏測試/ONNX 一致性/影子模式是部署期的事，不在此）：

  1. 黃金集 coverage @ precision ≥ 0.90，且不低於 baseline − 2pt（產品指標）
  2. 分組交叉驗證 macro-F1 不低於 baseline − 1σ（防單點運氣——
     GroupKFold 按 案場×日期，每折重訓探針再考黃金折，不是單點）
  3. 沒有任何類別 recall 掉到 0（防小類被犧牲換總分）

黃金答案 = G1 仲裁後的 data/golden/golden_labels.csv（fileId,cls），
由 src/g1_gate.py 的仲裁清單人工裁定後彙出。

    uv run src/g2_gate.py --candidate v36 --baseline v35

exit code 0 = 全過可上線；1 = 有檢查沒過。報告存 data/golden/g2_report.json。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import features as features_mod
import split as split_mod
from core.evaluation_metrics import PRECISION_BAR, coverage_at_precision

COVERAGE_SLACK = 0.02  # 不低於 baseline − 2pt
F1_SLACK_SIGMA = 1.0  # 不低於 baseline − 1σ


def probe(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)["clf"]


def corpus() -> tuple[list[str], object, dict]:
    """全語料 embedding＋split 標籤（分組 CV 的訓練側）。"""
    ids, emb = features_mod.load("siglip")
    idx = {f: i for i, f in enumerate(ids)}
    sp = json.loads((paths.SPLITS / f"{split_mod.current()}.json").read_text())
    return idx, emb, sp["labels"]


def golden(path: str) -> pd.DataFrame:
    g = pd.read_csv(path, dtype=str)
    assert {"fileId", "cls"} <= set(g.columns), "黃金答案需要 fileId,cls"
    return g.set_index("fileId")


def eval_probe(clf, X, y) -> dict:
    if not len(y):
        raise ValueError("eval_probe requires at least one labeled sample")
    y = np.asarray(y)
    p = clf.predict_proba(X)
    classes = list(clf.classes_)
    idx = p.argmax(1)
    pred = np.array([classes[i] for i in idx])
    conf = p[np.arange(len(p)), idx]
    correct = pred == y
    cov = coverage_at_precision(correct, conf, precision_bar=PRECISION_BAR)
    per_class = {}
    for c in sorted(set(y)):
        m = y == c
        rec = float(correct[m].mean())
        per_class[c] = {
            "n": int(m.sum()),
            "recall": round(rec, 3),
            "本輪可評": "✅" if m.sum() >= 30 else f"n={m.sum()} <30",
        }
    return {"top1": round(float(correct.mean()), 4), "coverage": round(cov, 4), "per_class": per_class}


def group_folds(golden_ids: list[str], labels: dict) -> list[set]:
    """黃金照片按 案場×日期 分組＝折。"""
    groups: dict[str, set] = {}
    fr = pd.read_csv(
        paths.FIELD_REPORTS / "raw" / "manifest.csv", dtype=str, keep_default_na=False, na_values=[""]
    )
    fr_map = dict(zip(fr.fileId, "樂氧森|" + fr.reportDate.astype(str)))
    daily = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
    d_map = dict(zip(daily.fileId, daily.constrId.astype(str) + "|" + daily.reportDate.astype(str)))
    for f in golden_ids:
        g = fr_map.get(f) or d_map.get(f) or "unknown"
        groups.setdefault(g, set()).add(f)
    return list(groups.values())


def grouped_cv(
    candidate_clf_cfg: tuple,
    baseline_clf_cfg: tuple,
    folds: list[set],
    idx: dict,
    emb,
    labels: dict,
    C: float,
) -> dict:
    """每折：全語料扣掉該折群組重訓探針（同超參數）→ 考該折的黃金照片。

    回 {model: [每折 macro-F1]}。兩顆模型用同一組折，差異才可比。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    out = {"candidate": [], "baseline": []}
    for fold in folds:
        tr_ids = [f for f in idx if f in labels and f not in fold]
        te_ids = [f for f in fold if f in idx and f in labels]
        if not tr_ids or not te_ids:
            continue
        Xtr = np.stack([emb[idx[f]] for f in tr_ids])
        ytr = np.array([labels[f] for f in tr_ids])
        Xte = np.stack([emb[idx[f]] for f in te_ids])
        yte = np.array([labels[f] for f in te_ids])
        for name, (clf_kind, pkl_path) in zip(
            ("candidate", "baseline"), (candidate_clf_cfg, baseline_clf_cfg)
        ):
            if clf_kind == "retrain":
                clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
                clf.fit(Xtr, ytr)
            else:
                clf = probe(pkl_path)
            pred = clf.predict(Xte)
            out[name].append(round(float(f1_score(yte, pred, average="macro", zero_division=0)), 4))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(paths.GOLDEN / "golden_labels.csv"))
    ap.add_argument("--candidate", required=True, help="候選模型（split 名，如 v36）")
    ap.add_argument("--baseline", required=True, help="現行模型（CURRENT 那版）")
    ap.add_argument("--cv", type=int, default=5, help="分組 CV 折數上限（群組少於此會自動縮）")
    a = ap.parse_args()

    g = golden(a.labels)
    c_path, b_path = (
        paths.MODELS / f"probe-siglip-{a.candidate}.pkl",
        paths.MODELS / f"probe-siglip-{a.baseline}.pkl",
    )
    for p in (c_path, b_path):
        if not p.exists():
            raise SystemExit(f"找不到 {p}")
    idx, emb, labels = corpus()

    keep = [f for f in g.index if f in idx]
    dropped = len(g) - len(keep)
    X = np.stack([emb[idx[f]] for f in keep])
    y = np.array([g.loc[f, "cls"] for f in keep])
    print(f"黃金集 {len(keep)} 張（{dropped} 張沒有 embedding，跳過）")

    report: dict = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "labels": a.labels,
        "candidate": a.candidate,
        "baseline": a.baseline,
        "golden_n": len(keep),
        "golden_dropped_no_emb": dropped,
        "checks": {},
    }
    verdicts: list[bool] = []

    cand_c, base_c = eval_probe(probe(c_path), X, y), eval_probe(probe(b_path), X, y)
    report["golden_eval"] = {"candidate": cand_c, "baseline": base_c}
    print(f"\n== ① 黃金集 coverage @ precision ≥ {PRECISION_BAR}")
    print(f"   candidate {a.candidate}: coverage {cand_c['coverage']} (top1 {cand_c['top1']})")
    print(f"   baseline  {a.baseline}: coverage {base_c['coverage']} (top1 {base_c['top1']})")
    ok1 = cand_c["coverage"] >= PRECISION_BAR and cand_c["coverage"] >= base_c["coverage"] - COVERAGE_SLACK
    print(f"   → {'PASS' if ok1 else 'FAIL'}")
    report["checks"]["coverage_at_precision"] = ok1
    verdicts.append(ok1)

    print("\n== ③ per-class recall（任何一類掉 0 = FAIL）")
    zeros = [c for c, s in cand_c["per_class"].items() if s["recall"] == 0 and s["n"] >= 5]
    small_zero = [c for c, s in cand_c["per_class"].items() if s["recall"] == 0 and s["n"] < 5]
    ok3 = not zeros
    print(f"   recall=0 的類（n≥5）：{zeros or '無'}；n<5 的 0 類（僅記錄）：{small_zero or '無'}")
    print(f"   → {'PASS' if ok3 else 'FAIL'}")
    report["checks"]["no_zero_recall"] = ok3
    report["zero_recall_small_n"] = small_zero
    verdicts.append(ok3)

    print(f"\n== ② 分組交叉驗證 macro-F1（{min(a.cv, 5)} 折，每折重訓 vs 現行權重）")
    folds = group_folds(keep, labels)[: a.cv]
    cv = grouped_cv(("retrain", None), ("pkl", b_path), folds, idx, emb, labels, C=300.0)
    cand_f1, base_f1 = np.array(cv["candidate"]), np.array(cv["baseline"])
    print(f"   candidate 各折：{cv['candidate']} → mean {cand_f1.mean():.4f} ± {cand_f1.std():.4f}")
    print(f"   baseline  各折：{cv['baseline']} → mean {base_f1.mean():.4f} ± {base_f1.std():.4f}")
    ok2 = bool(cand_f1.mean() >= base_f1.mean() - base_f1.std() * F1_SLACK_SIGMA)
    print(
        f"   → {'PASS' if ok2 else 'FAIL'}（門檻：不低於 baseline − 1σ = {base_f1.mean() - base_f1.std() * F1_SLACK_SIGMA:.4f}）"
    )
    report["checks"]["grouped_cv_macro_f1"] = ok2
    report["grouped_cv"] = {k: v for k, v in cv.items()}
    verdicts.append(ok2)

    report["verdict"] = "PASS" if all(verdicts) else "FAIL"
    out = paths.GOLDEN / "g2_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n總結：{report['verdict']}（{sum(verdicts)}/{len(verdicts)} 項通過）→ {out}")
    print("§7.2 其餘檢查（洩漏測試/ONNX 一致性/影子模式）屬部署期，不在本腳本。")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
