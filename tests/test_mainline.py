"""主線腳本端到端：訓練 → 評估 → 學習紀錄，以及工作台 CLI。

每日排程無人值守地跑這條鏈；它們以前沒有任何測試，壞了只會在早上的 Slack 通知裡變成「失敗步驟」。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import model_registry as registry
from core import paths
from core.labeler import load_reviews

CLASSES = ["油漆-塗裝", "泥作-打底", "防水-塗佈"]


def _version(name: str, seed: int = 0) -> list[str]:
    """三類、可分的假 embedding＋一份 split。回傳測試集 fileId。"""
    rng = np.random.default_rng(seed)
    centers = np.eye(3) * 5
    ids, emb, labels = [], [], {}
    for i in range(60):
        c = i % 3
        fid = f"p{i:03d}"
        ids.append(fid)
        emb.append(centers[c] + rng.normal(scale=0.6, size=3))
        labels[fid] = CLASSES[c]
    np.savez(registry.feature_path("siglip"), fileIds=np.array(ids), emb=np.array(emb, dtype=np.float32))
    train, test = ids[:45], ids[45:]
    registry.split_path(name).write_text(
        json.dumps({"name": name, "encoder": "siglip", "train": train, "test": test, "labels": labels}),
        encoding="utf-8",
    )
    man = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    man["dailyReportInfoId"] = ""  # evaluate 的工項融合要讀這欄
    man.to_csv(paths.MANIFEST, index=False)
    return test


def test_train_evaluate_journal_chain(pms_env):
    import evaluate
    import journal
    import train

    _version("v1")
    _, acc, (_, _, ids_te) = train.probe("v1", log=lambda *a: None)
    assert registry.probe_path("v1").exists() and acc > 0.9 and len(ids_te) == 15
    assert json.loads(registry.probe_path("v1").with_suffix(".json").read_text())["C"] == registry.PROBE_C

    out = Path(evaluate.run("v1", baseline=False, run_tag="v1", log=lambda *a: None))
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["support"] == 15 and metrics["top1"] == round(acc, 4)
    assert set(metrics["perClass"]) == set(CLASSES) and "workItemFusion" in metrics
    cfg = json.loads((out / "config.json").read_text())
    assert cfg == {
        "labelsVersion": cfg["labelsVersion"],
        "split": "v1",
        "encoder": "siglip",
        "model": "probe-siglip-v1.pkl",
    }

    _version("v2", seed=1)
    train.probe("v2", log=lambda *a: None)
    evaluate.run("v2", baseline=False, run_tag="v2", log=lambda *a: None)
    text = journal.build()
    assert "共 2 個 run" in text and "| v1 |" in text and "| v2 |" in text
    assert f"v2 vs {out.name}" in text  # 第二節比的是上一版，不是「無」
    assert journal.write().endswith("JOURNAL.md")


def test_journal_self_check():
    import journal

    journal.demo()  # 解析誤判檔名、慣犯、幽靈類別不入 macro——這些規則以前只在 --self-check 裡跑


def test_evaluate_refuses_missing_probe(pms_env):
    import evaluate

    _version("v1")
    with pytest.raises(FileNotFoundError):
        evaluate.run("v1", baseline=False, log=lambda *a: None)


def test_pms_cli_status_matches_routing_and_decide_writes_through(pms_env, route_queue, capsys):
    import pms

    route_queue(["a", "u1"])
    assert pms.main(["status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["needsReview"] == 2 and status["photos"] >= 2

    assert (
        pms.main(
            ["decide", "--id", "a", "--action", "classified", "--label", "泥作-打底", "--reviewer", "cli"]
        )
        == 0
    )
    assert load_reviews() == {"a": "泥作-打底"}
    capsys.readouterr()
    assert pms.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["needsReview"] == 1  # 人審完立刻離開佇列

    # 不在分類表的類名、沒寫原因的暫緩：拒收，回 2，不寫檔
    assert (
        pms.main(["decide", "--id", "b", "--action", "classified", "--label", "不存在", "--reviewer", "cli"])
        == 2
    )
    assert pms.main(["decide", "--id", "b", "--action", "uncertain", "--reviewer", "cli"]) == 2
    assert load_reviews() == {"a": "泥作-打底"}
