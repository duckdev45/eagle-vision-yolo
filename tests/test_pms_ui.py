"""PMS 操作台互動測試：全部寫入隔離 fixture，不使用真實照片裁決。"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from core import paths, pms_decisions, pms_review, routing
from core import pms_store as store
from core.labeler import load_reviews


def app(section: str) -> AppTest:
    st.cache_data.clear()
    return AppTest.from_string(
        f"from ui.pms_workbench import workbench\nworkbench({section!r})", default_timeout=15
    )


@pytest.mark.parametrize("section", ["photos", "candidates", "overview"])
def test_workbench_renders_without_a_model(pms_env, section):
    result = app(section).run()
    assert not result.exception
    assert not result.error
    assert not paths.REVIEW.exists()
    assert not store.database_path().exists()


@pytest.mark.parametrize("section", ["photos", "candidates", "overview"])
def test_workbench_empty_install(pms_env, section):
    paths.MANIFEST.unlink()
    result = app(section).run()
    assert not result.exception
    assert not result.error


def test_manual_photo_confirmation_in_ui(pms_env):
    result = app("photos").run()
    result.selectbox(key="pms_photo_filter").set_value("全部")  # 沒跑分流＝沒有「待複核」，看全部
    result.text_input(key="pms_reviewer").set_value("ui-tester")
    result.text_input(key="pms_photo_search").set_value("打底施作").run()
    assert not result.exception
    result.selectbox(key="pms_class_a").set_value("泥作-打底")
    result.text_input(key="pms_reason_a").set_value("看圖確認")
    next(button for button in result.button if button.label == "儲存判斷").click().run()
    assert not result.exception
    assert not result.error
    assert load_reviews() == {"a": "泥作-打底"}
    assert store.active_decisions()["a"]["reviewer"] == "ui-tester"


def test_defect_flag_toggle_in_ui_keeps_trade_verdict(pms_env):
    result = app("photos").run()
    result.selectbox(key="pms_photo_filter").set_value("全部")  # 沒跑分流＝沒有「待複核」，看全部
    result.text_input(key="pms_reviewer").set_value("ui-tester")
    result.text_input(key="pms_photo_search").set_value("打底施作").run()
    next(box for box in result.checkbox if box.label.startswith("這張是缺失改善照")).check()
    next(button for button in result.button if button.label == "儲存缺失旗標").click().run()
    assert not result.exception and not result.error
    assert store.latest("defect")["a"]["defect"] is True
    assert not paths.REVIEW.exists()  # 缺失旗標不寫 review.csv，工種裁決不動
    df, _ = pms_review.snapshot()
    row = df.set_index("fileId").loc["a"]
    assert bool(row.defectFlag) and row.defectSource == "human"


def test_candidate_approval_in_ui_does_not_auto_label(pms_env):
    key = pms_decisions.propose_candidate(
        "裝修-消音板", ["u1", "u2"], reviewer="tester", definition="板材孔洞"
    )
    result = app("candidates").run()
    result.text_input(key="pms_reviewer").set_value("ui-tester")
    next(field for field in result.text_area if field.label == "新類定義").set_value("可見消音板")
    next(field for field in result.text_area if field.label == "不包含的情況／容易混淆的類別").set_value(
        "一般木平板"
    )
    next(field for field in result.text_input if field.label == "分類樹或 QS 依據").set_value("現場分類表")
    next(button for button in result.button if button.label == "核准此新類").click().run()
    assert not result.exception
    assert not result.error
    assert store.latest("candidate")[key]["status"] == "approved"
    assert load_reviews() == {}


def test_real_entrypoint_routes_to_pms_workbench(pms_env):
    st.cache_data.clear()
    entry = Path(__file__).resolve().parents[1] / "src" / "app.py"
    result = AppTest.from_file(str(entry), default_timeout=15).run()
    assert not result.exception
    assert any(s.value == "收件匣" for s in result.subheader)  # 預設落在收件匣
    for section in ("總覽", "報告"):
        result.session_state["main_nav"] = section
        result.run()
        assert not result.exception
        assert not result.error
    result.session_state["main_nav"] = "進階"
    result.run()
    for name in ("照片工種", "新工種候選", "同步與重訓"):
        result.radio(key="adv_nav").set_value(name).run()
        assert not result.exception
        assert not result.error


def _route_with(pms_env, monkeypatch):
    import pandas as pd

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
    routing.build(log=lambda *a: None)


def test_inbox_without_routing_offers_to_route(pms_env):
    st.cache_data.clear()
    result = AppTest.from_string("from ui.inbox import inbox\ninbox()", default_timeout=15).run()
    assert not result.exception and not result.error
    assert any(b.label == "▶ 立即分流" for b in result.button)


def test_inbox_one_click_resolves_and_leaves_the_queue(pms_env, monkeypatch):
    _route_with(pms_env, monkeypatch)
    st.cache_data.clear()
    result = AppTest.from_string("from ui.inbox import inbox\ninbox()", default_timeout=15).run()
    assert not result.exception and not result.error
    result.text_input(key="pms_reviewer").set_value("ui-tester")
    rule_button = next(b for b in result.button if b.label.startswith("✓ 油漆-塗裝（規則"))
    rule_button.click().run()
    assert not result.exception and not result.error
    assert load_reviews() == {"b": "油漆-塗裝"}
    assert store.active_decisions()["b"]["reviewer"] == "ui-tester"
    assert "b" not in set(routing.queue_items().fileId)


def test_inbox_reviewer_defaults_from_env(pms_env, monkeypatch):
    monkeypatch.setenv("PMS_REVIEWER", "env-reviewer")
    _route_with(pms_env, monkeypatch)
    st.cache_data.clear()
    result = AppTest.from_string("from ui.inbox import inbox\ninbox()", default_timeout=15).run()
    assert result.text_input(key="pms_reviewer").value == "env-reviewer"
