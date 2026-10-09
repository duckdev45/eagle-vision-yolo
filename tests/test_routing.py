"""每日分流的規則（core/routing.py）。

每一條都對應 2026-10-09 定案的決定：門檻 0.9、自動桶抽 5%、自動確認不寫 review.csv、
模型訊號不能來自背過那張照片的模型。改規則要連這裡一起改。
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import paths, pms_review, routing
from core.labeler import load_reviews

TODAY = date(2026, 10, 9)
MODEL_CLASSES = {"泥作-打底", "泥作-粉光", "油漆-塗裝", "雜項-缺失改善"}


def _row(**kw) -> dict:
    base = {
        "fileId": "x",
        "ruleClass": "泥作-打底",
        "modelClass": "泥作-打底",
        "modelConfidence": 0.95,
        "stageSource": "model",
        "reportDate": "2026-10-01",
        "humanClass": "",
        "reviewState": "",
        "hasPhoto": True,
    }
    return {**base, **kw}


def _route(*rows: dict) -> pd.DataFrame:
    return routing.route_frame(pd.DataFrame(list(rows)), MODEL_CLASSES, TODAY)


def _not_audited(prefix: str = "p") -> str:
    """找一個不會被 5% 抽查抽到的 fileId，讓「自動確認」的案例穩定。"""
    return next(f"{prefix}{i}" for i in range(1000) if not routing.audit_pick(f"{prefix}{i}"))


def test_threshold_matches_the_decision():
    assert routing.AUTO_CONFIDENCE == 0.9
    assert routing.AUDIT_PERCENT == 5


def test_rule_and_model_agree_with_high_confidence_is_auto():
    fid = _not_audited()
    out = _route(_row(fileId=fid, modelConfidence=0.9))
    assert out.bucket.iloc[0] == "auto"
    assert out.reason.iloc[0] == ""


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"modelClass": "油漆-塗裝"}, "規則與模型不同"),
        ({"ruleClass": ""}, "標題沒有對應規則"),
        ({"ruleClass": "木作-天花封板", "modelClass": "木作-天花封板"}, "模型尚未涵蓋此類"),
        ({"modelConfidence": 0.89}, "模型信心不足"),
        ({"stageSource": "manual"}, "打底／粉光階段待人工"),
        ({"ruleClass": "雜項-缺失改善", "modelClass": "雜項-缺失改善"}, "缺失改善照：工種待看圖"),
    ],
)
def test_anything_contested_goes_to_the_human_queue(change, reason):
    out = _route(_row(fileId=_not_audited(), **change))
    assert out.bucket.iloc[0] == "queue"
    assert reason in out.reason.iloc[0]


def test_disagreement_is_not_also_flagged_as_low_confidence():
    """規則≠模型時信心低是理所當然，再標一次「信心不足」只是噪音。"""
    out = _route(_row(modelClass="油漆-塗裝", modelConfidence=0.4))
    assert out.reason.iloc[0] == "規則與模型不同"


def test_queue_priority_puts_disagreements_first():
    out = _route(
        _row(fileId="lowconf", modelConfidence=0.5),
        _row(fileId="fight", modelClass="油漆-塗裝"),
        _row(fileId="silent", ruleClass=""),
    ).sort_values("priority")
    assert list(out.fileId) == ["fight", "silent", "lowconf"]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"hasPhoto": False}, "沒有原圖"),
        ({"modelClass": "", "modelConfidence": float("nan")}, "尚無圖像特徵"),
        ({"reportDate": "2094-01-05"}, "日報日期在未來"),
        ({"reportDate": "not-a-date"}, "日報日期無法解析"),
    ],
)
def test_broken_inputs_are_quarantined_not_queued(change, reason):
    out = _route(_row(**change))
    assert out.bucket.iloc[0] == "quarantine"
    assert out.reason.iloc[0] == reason


def test_human_verdict_and_held_states_leave_routing():
    out = _route(
        _row(fileId="a", humanClass="油漆-塗裝", modelClass="泥作-粉光"),
        _row(fileId="b", reviewState="uncertain"),
        _row(fileId="c", reviewState="excluded", reportDate="2094-01-01"),
    )
    assert list(out.bucket) == ["done", "held", "held"]


def test_audit_sample_is_deterministic_and_about_five_percent():
    ids = [f"photo-{i}" for i in range(4000)]
    picked = [routing.audit_pick(f) for f in ids]
    assert picked == [routing.audit_pick(f) for f in ids]  # 明天重算，抽到的還是同一批
    assert 0.035 < sum(picked) / len(ids) < 0.065


def test_audited_auto_rows_go_to_humans_with_their_own_reason():
    fid = next(f"p{i}" for i in range(1000) if routing.audit_pick(f"p{i}"))
    out = _route(_row(fileId=fid))
    assert out.bucket.iloc[0] == "audit"
    assert out.reason.iloc[0] == "自動桶抽查"


def test_oof_never_scores_a_photo_with_a_model_that_saw_its_group():
    """同組（案場×日期）的照片絕對不能出現在預測它的那一折的訓練集裡。

    做法：每組給一個只屬於該組的特徵方向與獨有類別。若模型看過同組，就預測得出那個類別；
    out-of-fold 時那個類別根本不在訓練集，該欄機率必須是 0。
    """
    rng = np.random.default_rng(0)
    groups = [f"site|day{i}" for i in range(6) for _ in range(4)]
    y = [f"類別-{i}" for i in range(6) for _ in range(4)]
    X = np.vstack([np.eye(6)[i] * 5 + rng.normal(0, 0.01, 6) for i in range(6) for _ in range(4)])
    classes = sorted(set(y))
    proba = routing.oof_proba(X, y, groups, classes, folds=6, C=1.0)
    for row, label in zip(proba, y):
        assert row[classes.index(label)] == 0.0


def test_oof_needs_at_least_two_groups():
    with pytest.raises(ValueError):
        routing.oof_proba(np.ones((4, 2)), ["a", "b", "a", "b"], ["g"] * 4, ["a", "b"])


def test_stage_decided_by_title_uses_group_confidence():
    classes = ["油漆-塗裝", "泥作-打底", "泥作-粉光"]
    proba = np.array([[0.02, 0.5, 0.48]])
    [(pred, conf, source)] = routing._resolve(proba, classes, ["3F 打底施作"])
    assert (pred, source) == ("泥作-打底", "title")
    assert conf == pytest.approx(0.98)  # 圖像只判「打底粉光群」，群的總機率才是它的信心


def test_build_never_writes_review_csv_and_exports_queue(pms_env, monkeypatch):
    """分流只決定誰要看，不決定答案：自動確認的照片不得出現在 review.csv。"""
    signals = pd.DataFrame(
        {
            "fileId": ["a", "b", "u1"],
            "modelClass": ["泥作-打底", "泥作-打底", "泥作-打底"],
            "modelConfidence": [0.99, 0.99, 0.99],
            "stageSource": ["model"] * 3,
            "signal": ["oof", "oof", "live"],
        }
    )
    monkeypatch.setattr(
        routing, "model_signals", lambda log=print: (signals, {"model": "v9", "modelClasses": ["泥作-打底"]})
    )
    summary = routing.build(today=TODAY, log=lambda *a: None)
    assert not paths.REVIEW.exists()
    assert load_reviews() == {}
    counts = summary["counts"]
    assert counts["queue"] >= 1  # b 是油漆標題、模型說打底 → 規則與模型不同
    exported = json.loads((paths.QUEUE_EXPORT / "queue.json").read_text(encoding="utf-8"))
    assert exported["schemaVersion"] == 1 and exported["autoConfidence"] == 0.9
    assert {i["fileId"] for i in exported["items"]} == set(routing.queue_items().fileId)
    assert (paths.ROUTE / "latest.json").exists()
    assert paths.ROUTE_HISTORY.read_text(encoding="utf-8").count("\n") == 1


def test_resolving_from_the_inbox_goes_through_the_workbench_decision(pms_env, monkeypatch):
    signals = pd.DataFrame(
        {
            "fileId": ["b"],
            "modelClass": ["泥作-打底"],
            "modelConfidence": [0.99],
            "stageSource": ["model"],
            "signal": ["oof"],
        }
    )
    monkeypatch.setattr(
        routing, "model_signals", lambda log=print: (signals, {"model": "v9", "modelClasses": ["泥作-打底"]})
    )
    routing.build(today=TODAY, log=lambda *a: None)
    assert "b" in set(routing.queue_items().fileId)
    with pytest.raises(ValueError):
        routing.resolve("b", reviewer="", label="油漆-塗裝")  # 沒有確認者不收
    routing.resolve("b", reviewer="tester", label="油漆-塗裝")
    assert load_reviews()["b"] == "油漆-塗裝"
    assert "b" not in set(routing.queue_items().fileId)  # 人審完立刻離開收件匣，不必等明天重算


def test_audit_precision_counts_only_reviewed_samples(pms_env):
    pd.DataFrame({"fileId": ["a", "b", "u1"], "autoClass": ["泥作-打底", "油漆-塗裝", "泥作-打底"]}).to_csv(
        paths.ROUTE_AUDIT, index=False
    )
    pd.DataFrame(
        [
            {"fileId": "a", "cls": "泥作-打底", "note": "", "reviewedAt": "x", "box": ""},
            {"fileId": "b", "cls": "泥作-打底", "note": "", "reviewedAt": "x", "box": ""},
        ]
    ).to_csv(paths.REVIEW, index=False)
    assert routing.audit_precision() == {"sampled": 3, "reviewed": 2, "agree": 1, "precision": 0.5}


def test_every_view_reads_the_same_queue(pms_env, monkeypatch, capsys):
    """收件匣、工作台（pms-status）、make queue 對「誰要人看」只能有一個答案。"""
    import review as review_cli

    signals = pd.DataFrame(
        {
            "fileId": ["a", "b", "u1"],
            "modelClass": ["泥作-打底", "泥作-打底", "泥作-打底"],
            "modelConfidence": [0.99, 0.99, 0.5],
            "stageSource": ["model"] * 3,
            "signal": ["oof", "oof", "live"],
        }
    )
    monkeypatch.setattr(
        routing, "model_signals", lambda log=print: (signals, {"model": "v9", "modelClasses": ["泥作-打底"]})
    )
    routing.build(today=TODAY, log=lambda *a: None)
    inbox = set(routing.queue_items().fileId)
    assert inbox
    snap, _ = pms_review.snapshot(
        {"name": "v9", "classes": [], "scores": {}, "stage": {}, "train": [], "test": [], "warning": ""}
    )
    assert set(snap[snap.needsReview.astype(bool)].fileId) == inbox
    assert review_cli.main([]) == 0
    assert f"待看 {len(inbox)} 張" in capsys.readouterr().out
    reasons = dict(zip(routing.queue_items().fileId, routing.queue_items().reason))
    assert all(snap.set_index("fileId").reviewReason[f].startswith(reasons[f]) for f in inbox)


def test_resolve_keeps_or_replaces_evidence_boxes(pms_env):
    from core.labeler import load_boxes

    routing.resolve("b", reviewer="tester", label="油漆-塗裝", boxes=[[0, 0, 500, 500]])
    assert load_boxes()["b"] == [[0, 0, 500, 500]]
    routing.resolve("b", reviewer="tester", label="泥作-打底")  # 沒給框＝保留既有的框
    assert load_boxes()["b"] == [[0, 0, 500, 500]] and load_reviews()["b"] == "泥作-打底"
