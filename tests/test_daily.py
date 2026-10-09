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


# ── 編排主流程（src/daily.main）：排程無人值守，壞了只看得到紀錄，所以串接本身要測 ──────────
SUMMARY = {
    "model": "v9",
    "counts": {"auto": 5, "audit": 1, "queue": 2, "quarantine": 0},
    "auditPrecision": {},
}


@pytest.fixture()
def orchestra(pms_env, monkeypatch):
    """把每一步換成替身，記下呼叫順序；個別測試再把某一步換成會失敗的。"""
    import daily
    import features
    import prepare
    import sync
    from core import routing
    from pipeline import pms_workflow

    calls: list[str] = []
    monkeypatch.setattr(sync, "sync", lambda log=print: calls.append("sync"))
    monkeypatch.setattr(prepare, "run", lambda kind, log=print: calls.append("prepare"))
    monkeypatch.setattr(
        features, "extract", lambda model_key, src, log=print: calls.append(f"features-{model_key}")
    )
    monkeypatch.setattr(routing, "build", lambda log=print: calls.append("route") or SUMMARY)
    monkeypatch.setattr(daily, "training_debt", lambda: {"newReviews": 0, "newPhotos": 0, "daysSince": 1})
    monkeypatch.setattr(pms_workflow, "next_version", lambda: "v12")
    monkeypatch.setattr(
        pms_workflow, "run", lambda name, log=print, with_explain=True: calls.append(f"train-{name}")
    )

    def promote(name, *, export, log=print):
        assert export is daily.export_service_bundle  # 匯出由編排端注入，不是 core 自己找腳本
        calls.append(f"promote-{name}")
        return {"candidate": name, "baseline": "v9", "promoted": True, "reasons": []}

    monkeypatch.setattr(promotion, "promote", promote)
    return daily, calls


def _log_records() -> list[dict]:
    return [json.loads(line) for line in paths.DAILY_LOG.read_text(encoding="utf-8").splitlines()]


def test_daily_quiet_day_routes_without_training(orchestra, capsys):
    daily, calls = orchestra
    assert daily.main([]) == 0
    assert calls[:2] == ["sync", "prepare"] and calls[-1] == "route"
    assert not any(c.startswith("train") for c in calls)
    (record,) = _log_records()
    assert record["retrain"]["go"] is False and set(record["steps"].values()) == {"ok"}
    assert '"收件匣": 3' in capsys.readouterr().out  # queue + audit 是收件匣要人看的量


def test_daily_failed_step_does_not_stop_the_rest(orchestra, monkeypatch, capsys):
    """同步壞了（網路、帳密）照樣用本機資料分流；結束碼非 0 讓排程看得到。"""
    import sync

    daily, calls = orchestra

    def offline(log=print):
        raise ConnectionError("PMS 連不上")

    monkeypatch.setattr(sync, "sync", offline)
    assert daily.main([]) == 1
    assert "route" in calls and "prepare" in calls
    (record,) = _log_records()
    assert record["steps"]["sync"].startswith("failed: ConnectionError")
    assert record["steps"]["route"] == "ok"
    assert '"失敗步驟": ["sync"]' in capsys.readouterr().out


def test_daily_retrains_promotes_then_reroutes(orchestra, monkeypatch):
    daily, calls = orchestra
    monkeypatch.setattr(daily, "training_debt", lambda: {"newReviews": 25, "newPhotos": 0, "daysSince": 1})
    assert daily.main(["--no-sync"]) == 0
    assert "sync" not in calls
    assert calls[-3:] == ["train-v12", "promote-v12", "route"]  # 切換後收件匣改用新模型重算
    (record,) = _log_records()
    assert record["retrain"]["name"] == "v12" and record["promotion"]["promoted"] is True


def test_daily_failed_training_never_reaches_promotion(orchestra, monkeypatch):
    from pipeline import pms_workflow

    daily, calls = orchestra
    monkeypatch.setattr(daily, "training_debt", lambda: {"newReviews": 25, "newPhotos": 0, "daysSince": 1})

    def broken(name, log=print, with_explain=True):
        raise RuntimeError("特徵缺")

    monkeypatch.setattr(pms_workflow, "run", broken)
    assert daily.main(["--no-sync"]) == 1
    assert not any(c.startswith("promote") for c in calls)
    assert "promotion" not in _log_records()[0]


def test_daily_dry_run_touches_nothing(orchestra, monkeypatch):
    daily, calls = orchestra
    monkeypatch.setattr(daily, "training_debt", lambda: {"newReviews": 25, "newPhotos": 0, "daysSince": 1})
    assert daily.main(["--dry-run"]) == 0
    assert calls == ["route"]  # 不同步、不訓練、不切換
    assert not paths.DAILY_LOG.exists()
