"""PMS 操作台互動測試：全部寫入隔離 fixture，不使用真實照片裁決。"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import paths
from core import pms_review as review
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


def test_candidate_approval_in_ui_does_not_auto_label(pms_env):
    key = review.propose_candidate("裝修-消音板", ["u1", "u2"], reviewer="tester", definition="板材孔洞")
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
    for name in ("② 資料總覽", "③ 照片工種", "⑤ 新工種候選"):
        result.radio(key="pms_nav").set_value(name).run()
        assert not result.exception
        assert not result.error
