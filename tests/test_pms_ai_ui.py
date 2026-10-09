"""驗證照片頁與複核佇列的 AI 按鈕、建議展示與人工採用。"""

from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from core import paths
from core import pms_vision as vision
from core.labeler import load_reviews


@pytest.mark.parametrize("section", ["photos", "queue"])
def test_ai_buttons_save_suggestions_then_human_accepts(pms_env, monkeypatch, section):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("PMS_OPENAI_MODEL", "test-vision")
    paths.LABELS_YAML.write_text(
        paths.LABELS_YAML.read_text().replace("min_class_size: 2", "min_class_size: 1")
    )
    df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    df.loc[df.fileId == "a", "predWorkItem"] = "油漆"
    df.to_csv(paths.MANIFEST, index=False)
    st.cache_data.clear()
    module = (
        "from ui.pms_workbench import workbench\nworkbench('photos')"
        if section == "photos"
        else ("from ui.review_ui import review_queue\nreview_queue()")
    )
    result = AppTest.from_string(module, default_timeout=15).run()
    assert not result.exception
    prefix = "pms_photo_ai" if section == "photos" else "pms_queue_ai"
    result.multiselect(key=f"{prefix}_ids").set_value(["a"]).run()
    calls = []

    def handler(request):
        body = json.loads(request.content)
        context = json.loads(body["input"][0]["content"][1]["text"])["context"]
        calls.append(context["fileId"])
        prediction = {
            "decision": "existing",
            "label": "泥作-打底",
            "visualEvidence": "可見砂漿層",
            "titleEvidence": "標題描述打底",
            "titleRelation": "agrees",
            "reason": "一致",
            "alternatives": [],
            "candidateDefinition": "",
        }
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "model": "test-vision",
                "id": "test",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": json.dumps(prediction)}]}
                ],
            },
        )

    original = vision.classify
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(vision, "classify", lambda ids, **kwargs: original(ids, client=client, **kwargs))
        result.button(key=f"{prefix}_run").click().run()
    assert calls == ["a"] and not result.exception and not result.error
    assert load_reviews() == {}
    if section == "photos":
        result.text_input(key="pms_photo_search").set_value("打底施作")
        result.text_input(key="pms_reviewer").set_value("human-tester").run()
        accept_key = "pms_accept_a"
    else:
        result.text_input(key="pms_queue_reviewer").set_value("human-tester").run()
        accept_key = "pms_queue_accept_a"
    assert any("標題描述打底" in str(item.value) for item in result.markdown)
    result.button(key=accept_key).click().run()
    assert not result.exception and not result.error
    assert load_reviews() == {"a": "泥作-打底"}


def test_unconfigured_ui_never_calls_api(pms_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PMS_OPENAI_MODEL", raising=False)
    monkeypatch.setattr(vision, "classify", lambda *a, **k: pytest.fail("未設定不能呼叫 API"))
    st.cache_data.clear()
    result = AppTest.from_string("from ui.pms_workbench import workbench\nworkbench('photos')").run()
    assert result.button(key="pms_photo_ai_run").disabled
    assert not result.exception
