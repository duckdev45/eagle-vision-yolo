"""施作項目與出工照邊界：同名標題、人工裁決也不能讓出工照跨入分類。"""

from __future__ import annotations

import pandas as pd
import pytest

import paths
from core import pms_exchange as exchange
from core import pms_review as review
from core.labeler import Labeler, save_review
from core.pms_source import work_items


def add_sources():
    df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    row = df.iloc[0].to_dict()
    extra = [
        {**row, "fileId": "worker", "title": "", "source": "WORKFORCE", "tradeName": "公工"},
        {
            **row,
            "fileId": "worker-labelled",
            "title": "打底施作",
            "source": "WORKFORCE",
            "tradeName": "打石工",
        },
        {**row, "fileId": "work-demolition", "title": "打石工拆除施作", "source": "WORK_ITEM"},
        {**row, "fileId": "source-missing", "title": "打底施作", "source": ""},
        {**row, "fileId": "free-content", "title": "打底施作", "source": "FREE_CONTENT"},
    ]
    pd.concat([df, pd.DataFrame(extra)], ignore_index=True).to_csv(paths.MANIFEST, index=False)


def test_source_filter_keeps_work_items_including_demolition(pms_env):
    add_sources()
    before = paths.MANIFEST.read_bytes()
    pool = review.load_pool()
    assert "work-demolition" in set(pool.fileId)
    assert not {"worker", "worker-labelled", "source-missing", "free-content", "foreign"} & set(pool.fileId)
    assert set(pool.source) == {"WORK_ITEM"}
    assert paths.MANIFEST.read_bytes() == before


def test_missing_source_fails_without_guessing_from_title(pms_env):
    df = pd.read_csv(paths.MANIFEST).drop(columns="source")
    df.to_csv(paths.MANIFEST, index=False)
    with pytest.raises(ValueError, match="source"):
        review.load_pool()
    assert work_items(pd.DataFrame()).empty


@pytest.mark.parametrize("operation", ["export", "decide", "candidate"])
def test_workforce_cannot_enter_review_operations(pms_env, operation):
    add_sources()
    with pytest.raises(ValueError, match="WORK_ITEM"):
        if operation == "export":
            exchange.export_packet(["a", "worker"])
        elif operation == "decide":
            review.decide("worker", "classified", reviewer="tester", label="泥作-打底")
        else:
            review.propose_candidate("拆除-打石", ["worker-labelled"], reviewer="tester", definition="測試")


def test_training_and_review_share_source_filter_even_with_old_override(pms_env):
    import newclass
    import review as review_cli
    from labels import labeled_manifest, pending_classes, unclaimed
    from ui import data as ui_data

    add_sources()
    save_review("worker-labelled", "泥作-打底")
    raw = paths.MANIFEST.read_bytes()
    # 小類門檻降低，使被排除的原因只能是來源，而非不足樣本。
    paths.LABELS_YAML.write_text(
        paths.LABELS_YAML.read_text().replace("min_class_size: 2", "min_class_size: 1")
    )
    ui_data._manifest.clear()
    for df in (labeled_manifest(), review_cli._labeled()[0], ui_data.labeled()):
        assert "worker-labelled" not in set(df.fileId)
        assert set(df.source) == {"WORK_ITEM"}
    assert pending_classes().empty
    assert "worker" not in set(unclaimed().fileId)
    _, recognized, _ = newclass.pools("pms")
    assert recognized["打底施作"] == 1
    assert paths.MANIFEST.read_bytes() == raw


def test_new_split_never_includes_workforce_override_or_crop(pms_env, monkeypatch):
    import split as split_mod

    add_sources()
    save_review("worker-labelled", "泥作-打底", boxes=[[0, 0, 200, 200]])
    monkeypatch.setattr(
        Labeler,
        "load",
        classmethod(
            lambda cls, path=None: cls(
                {
                    **pms_env["cfg"],
                    "min_class_size": 1,
                }
            )
        ),
    )
    payload = split_mod.build(
        "only-work-items", with_legacy=False, legacy_fill=0, min_train=0, log=lambda *a: None
    )
    assert "worker-labelled" not in payload["labels"]
    assert "worker-labelled#0" not in payload["train"]
    assert not {"worker", "source-missing", "free-content"} & set(payload["train"] + payload["test"])
