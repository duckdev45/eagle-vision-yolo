"""下載壞圖補抓與品質盤點的回歸測試。"""

from io import BytesIO
from random import Random

import pandas as pd
from PIL import Image


def _valid_jpeg() -> bytes:
    buf = BytesIO()
    Image.frombytes("RGB", (128, 128), Random(0).randbytes(128 * 128 * 3)).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def test_photo_quality_rejects_large_non_image_and_repairs_atomically(tmp_path, monkeypatch):
    import paths
    import sync
    from photo_quality import bytes_problem, file_problem

    photos = tmp_path / "raw" / "photos"
    photos.mkdir(parents=True)
    monkeypatch.setattr(paths, "RAW", tmp_path / "raw")
    monkeypatch.setattr(paths, "PHOTOS", photos)
    broken = photos / "bad.jpg"
    broken.write_bytes(b"not-a-photo" * 200)
    assert file_problem(broken) == "decode_error"
    assert bytes_problem(broken.read_bytes()) == "decode_error"

    good = _valid_jpeg()
    assert bytes_problem(good) is None
    sync._store_valid_photo("bad", ".jpg", good, [broken])
    assert broken.read_bytes() == good
    assert next((tmp_path / "raw" / "quarantine").glob("bad-*.jpg")).read_bytes() == b"not-a-photo" * 200
    assert not (photos / ".bad.jpg.download").exists()


def test_quality_audit_marks_only_train_relevant_missing_photo(tmp_path, monkeypatch):
    import labels
    import paths
    import pms_quality

    photos = tmp_path / "photos"
    images = tmp_path / "images"
    photos.mkdir()
    images.mkdir()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"fileId": "train", "source": "WORK_ITEM", "active": "True"},
            {"fileId": "other", "source": "WORK_ITEM", "active": "True"},
        ]
    ).to_csv(manifest, index=False)
    (photos / "other.jpg").write_bytes(b"broken")
    monkeypatch.setattr(paths, "MANIFEST", manifest)
    monkeypatch.setattr(paths, "PHOTOS", photos)
    monkeypatch.setattr(paths, "IMAGES", images)
    monkeypatch.setattr(labels, "labeled_manifest", lambda: pd.DataFrame({"fileId": ["train"]}))

    issues, summary = pms_quality.audit()
    assert set(issues.fileId) == {"train", "other"}
    assert summary["trainingRelevantIssues"] == 1
    assert summary["rawProblems"] == {"missing": 1, "too_small:6": 1}


def test_sync_refetches_existing_bad_photo(pms_env, monkeypatch):
    import paths
    import sync

    good = _valid_jpeg()
    bad = paths.PHOTOS / "a.jpg"
    original = bad.read_bytes()

    class FakePms:
        last_total = 1

        def login(self):
            pass

        def list_reports(self, **kwargs):
            yield {"dailyReportInfoId": "r1", "version": 1, "status": "SUBMITTED", "constrName": "s1"}

        def get_report(self, report_id):
            return {
                "dailyReportInfoId": report_id,
                "constrId": "s1",
                "reportDate": "2026-08-01",
                "status": "SUBMITTED",
                "pages": [
                    {
                        "contentKind": "WORK_ITEM",
                        "content": {
                            "title": "打底",
                            "photos": [{"id": "a", "name": "a.jpg", "type": "image/jpeg", "url": "signed"}],
                        },
                    }
                ],
            }

        def download(self, url):
            assert url == "signed"
            return good

        def close(self):
            pass

    monkeypatch.setattr(sync, "Pms", FakePms)
    result = sync.sync(log=lambda *_: None)

    assert result["downloaded"] == 1 and result["failed"] == 0
    assert bad.read_bytes() == good
    assert next((paths.RAW / "quarantine").glob("a-*.jpg")).read_bytes() == original
