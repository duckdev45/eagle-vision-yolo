"""圖像與標題請求、API 失敗、來源隔離及人工真值邊界。全部使用 MockTransport。"""

from __future__ import annotations

import base64
import io
import json

import httpx
import pandas as pd
import pytest
from PIL import Image

from core import paths, pms_decisions, pms_photos
from core import pms_store as store
from core import pms_vision as vision
from core.labeler import load_reviews


@pytest.fixture()
def configured(pms_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-log")
    monkeypatch.setenv("PMS_OPENAI_MODEL", "test-vision-model")
    return pms_env


def answer(**changes):
    result = {
        "decision": "existing",
        "label": "泥作-打底",
        "visualEvidence": "可見砂漿底層",
        "titleEvidence": "標題記打底施作",
        "titleRelation": "agrees",
        "reason": "圖片與標題相符",
        "alternatives": [],
        "candidateDefinition": "",
    }
    result.update(changes)
    return result


def api_body(result=None):
    return {
        "id": "resp-test",
        "model": "test-vision-snapshot",
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(result or answer())}]}
        ],
    }


def test_request_contains_actual_image_title_catalog_and_no_human_truth(configured):
    requests = []

    def handler(request):
        assert str(request.url) == vision.ENDPOINT
        body = json.loads(request.content)
        requests.append(body)
        parts = body["input"][0]["content"]
        assert parts[0]["type"] == "input_image"
        data = base64.b64decode(parts[0]["image_url"].split(",", 1)[1])
        assert Image.open(io.BytesIO(data)).format == "JPEG"
        context = json.loads(parts[1]["text"])
        assert context["context"]["title"] == "打底施作"
        assert context["context"]["source"] == "WORK_ITEM"
        assert "humanClass" not in context["context"] and "modelClass" not in context["context"]
        assert set(context["catalog"]) == set(pms_photos.catalog())
        assert body["store"] is False
        assert body["text"]["format"]["strict"] is True
        return httpx.Response(200, json=api_body())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = vision.classify(["a"], client=client)
        second = vision.classify(["a"], client=client)
    assert first["completed"] == 1 and not first["failed"]
    assert second["cached"] == 1 and len(requests) == 1
    assert load_reviews() == {} and store.active_decisions() == {}
    suggestion = next(iter(store.latest("suggestion").values()))
    assert suggestion["titleEvidence"] == "標題記打底施作"
    assert suggestion["titleRelation"] == "agrees"
    assert suggestion["model"] == "test-vision-snapshot"
    metadata = next(iter(store.latest("vision_cache").values()))
    assert metadata["responseId"] == "resp-test"
    assert "test-secret" not in json.dumps(store.events("vision_cache"))
    pms_decisions.decide(
        "a", "classified", reviewer="tester", label=suggestion["label"], proposal_id=suggestion["proposalId"]
    )
    assert load_reviews() == {"a": "泥作-打底"}


@pytest.mark.parametrize("changed", ["image", "title", "catalog", "model", "force"])
def test_cache_changes_with_inputs_or_explicit_rerun(configured, changed):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=api_body())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert vision.classify(["a"], client=client)["completed"] == 1
        if changed == "image":
            Image.new("RGB", (60, 40), (30, 100, 200)).save(paths.PHOTOS / "a.jpg")
        elif changed == "title":
            df = pd.read_csv(paths.MANIFEST, keep_default_na=False)
            df.loc[df.fileId == "a", "title"] = "另一個標題"
            df.to_csv(paths.MANIFEST, index=False)
        elif changed == "catalog":
            paths.LABELS_YAML.write_text(paths.LABELS_YAML.read_text() + "\n# new catalog revision\n")
        result = vision.classify(
            ["a"], client=client, force=changed == "force", model="other-model" if changed == "model" else ""
        )
    assert result["completed"] == 1 and not result["failed"] and len(requests) == 2


@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        "connection",
        "401",
        "429",
        "500",
        "incomplete",
        "refusal",
        "invalid_json",
        "unknown_label",
        "missing_evidence",
        "bad_type",
    ],
)
def test_bad_response_never_becomes_a_label(configured, failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("test-secret-never-log", request=request)
        if failure == "connection":
            raise httpx.ConnectError("test-secret-never-log", request=request)
        if failure.isdigit():
            return httpx.Response(int(failure), text="test-secret-never-log")
        body = api_body()
        if failure == "incomplete":
            body["status"] = "incomplete"
        elif failure == "refusal":
            body["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
        elif failure == "invalid_json":
            body["output"][0]["content"][0]["text"] = "not json"
        elif failure == "unknown_label":
            body = api_body(answer(label="不存在-工種"))
        elif failure == "missing_evidence":
            body = api_body(answer(visualEvidence=""))
        elif failure == "bad_type":
            body = api_body(answer(titleRelation=[]))
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = vision.classify(["a"], client=client)
    assert result["completed"] == 0 and len(result["failed"]) == 1
    assert not store.latest("suggestion") and not load_reviews()
    assert "test-secret-never-log" not in json.dumps(result)


def test_failure_preserves_success_and_rerun_only_retries_failed_photo(configured):
    calls = []
    fail = True

    def handler(request):
        nonlocal fail
        context = json.loads(json.loads(request.content)["input"][0]["content"][1]["text"])["context"]
        fid = context["fileId"]
        calls.append(fid)
        if fid == "b" and fail:
            fail = False
            return httpx.Response(500)
        return httpx.Response(200, json=api_body())

    events = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = vision.classify(["a", "b"], client=client, progress=events.append)
        second = vision.classify(["a", "b"], client=client)
    assert first["completed"] == 1 and first["failed"][0]["fileId"] == "b"
    assert second["completed"] == 1 and second["cached"] == 1
    assert calls == ["a", "b", "b"] and events[-1]["done"] == 2
    assert load_reviews() == {}


def test_source_change_is_rechecked_before_saving_response(configured):
    def handler(request):
        df = pd.read_csv(paths.MANIFEST, keep_default_na=False)
        df.loc[df.fileId == "a", "source"] = "WORKFORCE"
        df.to_csv(paths.MANIFEST, index=False)
        return httpx.Response(200, json=api_body())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = vision.classify(["a"], client=client)
    assert result["failed"] and not store.latest("suggestion")


def test_workforce_is_blocked_before_any_api_request(configured):
    df = pd.read_csv(paths.MANIFEST, keep_default_na=False)
    df.loc[df.fileId == "b", "source"] = "WORKFORCE"
    df.to_csv(paths.MANIFEST, index=False)
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("不可傳送出工照"))) as client:
        with pytest.raises(ValueError, match="WORK_ITEM"):
            vision.classify(["a", "b"], client=client)
    assert not store.database_path().exists()


def test_missing_configuration_stops_before_writes(pms_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PMS_OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        vision.classify(["a"])
    assert not store.database_path().exists()


@pytest.mark.parametrize("decision", ["new_candidate", "uncertain", "not_construction"])
def test_nonexisting_results_are_suggestions_not_truth(configured, decision):
    result = answer(
        decision=decision,
        label="裝修-消音板" if decision == "new_candidate" else "",
        candidateDefinition="可見孔洞板材" if decision == "new_candidate" else "",
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=api_body(result)))
    ) as client:
        assert vision.classify(["a"], client=client)["completed"] == 1
    assert not load_reviews() and not store.latest("candidate") and not store.active_decisions()
