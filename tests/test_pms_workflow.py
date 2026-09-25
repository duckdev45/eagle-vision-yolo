"""PMS 來源、人工回流、候選生命週期及審閱包契約的回歸測試。"""

from __future__ import annotations

import copy
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import paths
from core import pms_exchange as exchange
from core import pms_review as review
from core import pms_store as store
from core.labeler import Labeler, load_boxes, load_reviews, orphan_reviews, save_review


def model():
    return {
        "name": "test",
        "classes": ["泥作-打底"],
        "scores": {"a": ("泥作-打底", 0.9, 0.8)},
        "train": ["a"],
        "test": ["b"],
        "warning": "",
    }


def response(ids):
    _, blob = exchange.export_packet(ids)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        out = json.loads(z.read("response-template.json"))
        assert set(json.loads(z.read("catalog.json"))) == set(review.catalog())
        assert Image.open(io.BytesIO(z.read(f"images/{ids[0]}.jpg"))).format == "JPEG"
    out["model"] = "test-reviewer"
    for row in out["results"]:
        row.update(decision="existing", label="泥作-打底", visualEvidence="測試用可見證據", reason="測試判斷")
    return out


def test_pms_pool_keeps_unknown_and_small_classes_read_only(pms_env):
    before = sorted(str(p) for p in pms_env["root"].rglob("*") if p.is_file())
    df, _ = review.snapshot(model())
    assert set(df.fileId) == {"a", "b", "c", "d", "u1", "u2"}
    indexed = df.set_index("fileId")
    assert indexed.loc["b", "route"] == "known_untrained"
    assert indexed.loc["c", "route"] == "unknown"
    assert indexed.loc["a", "part"] == "train"
    assert indexed.loc["b", "part"] == "test"
    assert indexed.loc["u1", "part"] == "unseen"
    assert review.class_inventory(df, model()).set_index("類別").loc["油漆-塗裝", "還差"] == 1
    assert before == sorted(str(p) for p in pms_env["root"].rglob("*") if p.is_file())


def test_import_is_suggestion_only_and_idempotent(pms_env):
    payload = response(["a", "b"])
    assert exchange.import_suggestions(payload) == 2
    assert exchange.import_suggestions(payload) == 0
    assert load_reviews() == {}
    assert store.active_decisions() == {}
    proposal = next(row for row in store.latest("suggestion").values() if row["fileId"] == "a")
    review.decide("a", "classified", reviewer="tester", label="泥作-打底", proposal_id=proposal["proposalId"])
    assert load_reviews() == {"a": "泥作-打底"}
    assert store.active_decisions()["a"]["proposalId"] == proposal["proposalId"]


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "extra", "unknown_class", "hash", "bad_decision"]
)
def test_invalid_batch_never_partially_imports(pms_env, mutation):
    payload = response(["a", "b"])
    if mutation == "duplicate":
        payload["results"][1] = copy.deepcopy(payload["results"][0])
    elif mutation == "missing":
        payload["results"].pop()
    elif mutation == "extra":
        payload["results"][1]["fileId"] = "other"
    elif mutation == "unknown_class":
        payload["results"][1]["label"] = "未知-類別"
    elif mutation == "hash":
        payload["results"][1]["imageHash"] = "wrong"
    else:
        payload["results"][1]["decision"] = []
    with pytest.raises(ValueError):
        exchange.import_suggestions(payload)
    assert store.latest("suggestion") == {}
    assert load_reviews() == {}


@pytest.mark.parametrize("changed", ["catalog", "context", "image"])
def test_changed_inputs_reject_response(pms_env, changed):
    payload = response(["a"])
    if changed == "catalog":
        paths.LABELS_YAML.write_text(paths.LABELS_YAML.read_text() + "\n# new revision\n")
    elif changed == "context":
        df = pd.read_csv(paths.MANIFEST, keep_default_na=False)
        df.loc[df.fileId == "a", "title"] = "油漆施作"
        df.to_csv(paths.MANIFEST, index=False)
    else:
        Image.new("RGB", (60, 40), (220, 20, 20)).save(paths.PHOTOS / "a.jpg")
    with pytest.raises(ValueError):
        exchange.import_suggestions(payload)
    assert store.latest("suggestion") == {}


def test_proposal_revalidated_when_confirmed(pms_env):
    exchange.import_suggestions(response(["a"]))
    proposal = next(iter(store.latest("suggestion").values()))
    Image.new("RGB", (60, 40), (20, 220, 20)).save(paths.PHOTOS / "a.jpg")
    with pytest.raises(ValueError):
        review.decide(
            "a", "classified", reviewer="tester", label="泥作-打底", proposal_id=proposal["proposalId"]
        )
    assert not paths.REVIEW.exists()


def test_manual_override_preserves_boxes_and_pause_can_be_reversed(pms_env):
    save_review("a", "泥作-打底", boxes=[[10, 20, 300, 400]])
    review.decide("a", "classified", reviewer="tester", label="油漆-塗裝")
    assert load_boxes()["a"] == [[10, 20, 300, 400]]
    review.decide("a", "uncertain", reviewer="tester", reason="材料被遮住")
    assert "a" not in set(Labeler.load().apply(review.load_pool(), drop_small=False).fileId)
    assert "a" in set(Labeler.load().apply(review.load_pool(), drop_small=False, overrides={}).fileId)
    save_review("a", "泥作-打底")
    assert "a" not in store.blocked_ids()
    assert "a" in set(Labeler.load().apply(review.load_pool(), drop_small=False).fileId)


def test_stale_screen_does_not_overwrite_new_decision(pms_env):
    revision = review.revision("a")
    review.decide("a", "classified", reviewer="first", label="泥作-打底")
    with pytest.raises(ValueError):
        review.decide("a", "classified", reviewer="second", label="油漆-塗裝", expected_revision=revision)
    assert load_reviews()["a"] == "泥作-打底"


def test_candidate_approval_defines_class_without_automatic_labels(pms_env):
    original_yaml = paths.LABELS_YAML.read_bytes()
    key = review.propose_candidate(
        "裝修-消音板", ["u1", "u2"], reviewer="tester", definition="可見板材與孔洞"
    )
    assert {"u1", "u2"} <= store.blocked_ids()
    seq = store.latest("candidate")[key]["_seq"]
    with pytest.raises(ValueError):
        review.resolve_candidate(
            key, approve=True, reviewer="tester", definition="板材", excludes="", basis="", expected_seq=seq
        )
    review.resolve_candidate(
        key,
        approve=True,
        reviewer="tester",
        definition="可見消音板",
        excludes="非一般平板",
        basis="現場分類表",
        expected_seq=seq,
    )
    assert "裝修-消音板" in review.catalog()
    assert load_reviews() == {}
    assert {"u1", "u2"} <= store.blocked_ids()
    for fid in ("u1", "u2"):
        review.decide(fid, "classified", reviewer="tester", label="裝修-消音板")
    assert set(Labeler.load().apply(review.load_pool()).fileId) == {"u1", "u2"}
    assert not orphan_reviews()
    assert paths.LABELS_YAML.read_bytes() == original_yaml


def test_known_untrained_cannot_be_proposed_as_new(pms_env):
    with pytest.raises(ValueError):
        review.propose_candidate("油漆-塗裝", ["b"], reviewer="tester", definition="油漆")
    assert not store.latest("candidate")


def test_reject_candidate_retains_history(pms_env):
    key = review.propose_candidate("裝修-消音板", ["u1"], reviewer="tester", definition="候選")
    seq = store.latest("candidate")[key]["_seq"]
    review.propose_candidate("裝修-消音板", ["u2"], reviewer="tester", definition="補照片")
    with pytest.raises(ValueError):
        review.resolve_candidate(
            key, approve=False, reviewer="tester", definition="", excludes="", basis="", expected_seq=seq
        )
    review.resolve_candidate(
        key,
        approve=False,
        reviewer="tester",
        definition="不開類",
        excludes="",
        basis="",
        expected_seq=store.latest("candidate")[key]["_seq"],
    )
    assert store.latest("candidate")[key]["status"] == "rejected"
    assert len(store.events("candidate")) == 3
    assert "裝修-消音板" not in review.catalog()


def test_calibration_uses_raw_rules_not_overridden_answer(pms_env):
    review.decide("a", "classified", reviewer="tester", label="油漆-塗裝")
    result = review.calibration()
    assert result["reviewed"] == 1 and result["ruleMatches"] == 0
    assert result["ruleAccuracy"] == 0


def test_discovery_only_groups_active_pms_unknowns(pms_env):
    df, _ = review.snapshot(model())
    groups = review.discover(df)
    assert groups and set(groups[0]["fileIds"]) == {"u1", "u2"}
    assert groups[0]["sites"] == 2 and groups[0]["days"] == 2


def test_pipeline_uses_pms_split_and_records_catalog(pms_env, monkeypatch):
    import split as split_mod
    from pipeline import pms_workflow

    calls = []
    payload = {
        "train": ["a", "b"],
        "test": ["u1"],
        "labels": {"a": "泥作-打底", "b": "油漆-塗裝", "u1": "泥作-打底"},
        "datasets": {"a": "pms", "b": "pms", "u1": "pms"},
    }

    def build(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(payload)

    monkeypatch.setattr(split_mod, "build", build)
    with pytest.raises(ValueError, match="特徵"):
        pms_workflow.build_split("pms-test")
    np.savez(paths.FEATURES / "siglip.npz", fileIds=np.array(["a", "b", "u1"]), emb=np.ones((3, 2)))
    result = pms_workflow.build_split("pms-test")
    assert calls[-1]["legacy_fill"] == 0 and calls[-1]["with_legacy"] is False
    assert result["pmsCatalogVersion"] == review.catalog_version()
    assert not (paths.SPLITS / "CURRENT").exists()
    with pytest.raises(ValueError):
        pms_workflow.check_new_run("pms-test")


def test_pipeline_stops_after_failed_prepare(pms_env, monkeypatch):
    import features
    import prepare
    from pipeline import pms_workflow

    monkeypatch.setattr(prepare, "run", lambda **kw: {"failed": 1})
    monkeypatch.setattr(features, "extract", lambda **kw: pytest.fail("失敗後不應繼續抽特徵"))
    with pytest.raises(RuntimeError):
        pms_workflow.run("pms-fail")
    assert not (paths.SPLITS / "CURRENT").exists()


@pytest.mark.parametrize("label", ["no-dash-extra", "../x-y", "工種-", "工\n種-內容", [], None])
def test_bad_candidate_labels_rejected(pms_env, label):
    with pytest.raises(ValueError):
        store.validate_label(label)
