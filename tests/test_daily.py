"""無人值守的那一半：重訓觸發、版本命名、自動切換的考卷門檻（src/daily.py、core/promotion.py）。

2026-10-09 定案：每日在本機排程跑、考卷過關就自動切換。門檻寫死在這裡，改了要連測試一起改。
"""

from __future__ import annotations

import json

import pytest

from core import paths, promotion


def _score(top1: float, f1: float, recall: dict[str, tuple[int, float]]) -> dict:
    return {
        "n": 200,
        "top1": top1,
        "macroF1": f1,
        "recall": {c: {"support": s, "recall": r} for c, (s, r) in recall.items()},
    }


def _exam(cand: dict, base: dict, n: int = 200) -> dict:
    return {"examSize": n, "candidateScore": cand, "baselineScore": base}


BASE = _score(0.86, 0.77, {"泥作-打底": (40, 0.9), "油漆-塗裝": (30, 0.8), "雜項-清潔": (3, 1.0)})


def test_equal_or_better_candidate_passes():
    ok, why = promotion.verdict(_exam(BASE, BASE))
    assert ok and why == []


def test_within_tolerance_is_still_non_inferior():
    cand = _score(0.851, 0.761, {"泥作-打底": (40, 0.9), "油漆-塗裝": (30, 0.8), "雜項-清潔": (3, 1.0)})
    assert promotion.verdict(_exam(cand, BASE))[0]


@pytest.mark.parametrize("metric", ["top1", "macroF1"])
def test_worse_headline_metric_blocks(metric):
    cand = json.loads(json.dumps(BASE))
    cand[metric] -= 0.02
    ok, why = promotion.verdict(_exam(cand, BASE))
    assert not ok and any(metric in w for w in why)


def test_a_sacrificed_class_blocks_even_when_totals_hold():
    cand = _score(0.87, 0.78, {"泥作-打底": (40, 0.95), "油漆-塗裝": (30, 0.6), "雜項-清潔": (3, 1.0)})
    ok, why = promotion.verdict(_exam(cand, BASE))
    assert not ok and any("油漆-塗裝" in w for w in why)


def test_tiny_classes_do_not_decide():
    cand = _score(0.86, 0.77, {"泥作-打底": (40, 0.9), "油漆-塗裝": (30, 0.8), "雜項-清潔": (3, 0.0)})
    assert promotion.verdict(_exam(cand, BASE))[0]


def test_small_exam_is_not_evidence():
    ok, why = promotion.verdict(_exam(BASE, BASE, n=promotion.MIN_EXAM - 1))
    assert not ok and "證據不足" in why[0]


def test_catalog_change_blocks():
    ok, why = promotion.verdict(_exam(BASE, BASE), catalog_ok=False)
    assert not ok and "分類表" in why[0]


def test_empty_score_is_json_safe():
    assert json.dumps(promotion.score([], []))  # NaN 會讓紀錄檔變成非法 JSON


def test_promote_switches_logs_and_exports_only_when_passed(pms_env, monkeypatch):
    from core import model_registry as registry

    registry.set_current("v1")
    monkeypatch.setattr(
        registry, "load_split", lambda name: {"pmsCatalogVersion": promotion.pms_photos.catalog_version()}
    )
    exported: list[str] = []

    monkeypatch.setattr(promotion, "exam", lambda c, b: _exam(BASE, BASE))
    rec = promotion.promote("v2", export=exported.append, log=lambda *a: None)
    assert rec["promoted"] and registry.current() == "v2" and exported == ["v2"]

    worse = _score(0.5, 0.4, {"泥作-打底": (40, 0.5), "油漆-塗裝": (30, 0.5), "雜項-清潔": (3, 1.0)})
    monkeypatch.setattr(promotion, "exam", lambda c, b: _exam(worse, BASE))
    rec = promotion.promote("v3", export=exported.append, log=lambda *a: None)
    assert not rec["promoted"] and registry.current() == "v2" and exported == ["v2"]

    log = [json.loads(line) for line in paths.PROMOTION_LOG.read_text(encoding="utf-8").splitlines()]
    assert [r["promoted"] for r in log] == [True, False]


def test_failed_export_does_not_hide_the_switch(pms_env, monkeypatch):
    from core import model_registry as registry

    registry.set_current("v1")
    monkeypatch.setattr(
        registry, "load_split", lambda name: {"pmsCatalogVersion": promotion.pms_photos.catalog_version()}
    )
    monkeypatch.setattr(promotion, "exam", lambda c, b: _exam(BASE, BASE))

    def boom(name):
        raise OSError("disk full")

    rec = promotion.promote("v2", export=boom, log=lambda *a: None)
    assert rec["promoted"] and "disk full" in rec["serviceBundleError"]


@pytest.mark.parametrize(
    ("reviews", "photos", "days", "go"),
    [
        (20, 0, 0, True),
        (19, 49, 30, False),
        (0, 50, 7, True),
        (0, 500, 6.9, False),
        (0, 0, 0, False),
    ],
)
def test_retrain_trigger(reviews, photos, days, go):
    import daily

    assert daily.should_retrain(reviews, photos, days)[0] is go


def test_next_version_skips_every_used_number(pms_env):
    from pipeline import pms_workflow

    paths.SPLITS.mkdir(parents=True, exist_ok=True)
    (paths.SPLITS / "v7.json").write_text("{}")
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    (paths.MODELS / "probe-so400m-v9.pkl").write_bytes(b"")
    (paths.REPORTS_OUT / "2026-10-01-v11").mkdir(parents=True)  # 訓練到一半失敗留下的報告
    (paths.SPLITS / "qms-v99.json").write_text("{}")  # 不是 vNN，不算
    name = pms_workflow.next_version()
    assert name == "v12"
    pms_workflow.check_new_run(name)


def test_unattended_training_can_skip_the_slow_explain_step(pms_env):
    from pipeline import pms_workflow

    titles = [t for t, _ in pms_workflow.training_steps("v1", with_explain=False)]
    assert "更新圖像解釋" not in titles
    assert "更新圖像解釋" in [t for t, _ in pms_workflow.training_steps("v1")]
