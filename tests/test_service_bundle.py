"""服務包匯出：每日排程考卷過關後會自動跑這條，沒人盯著，壞了要由測試先發現。"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from core import model_registry as registry
from core import paths, pms_review


def _trained_version(name: str = "v9") -> None:
    from sklearn.linear_model import LogisticRegression

    classes = ["油漆-塗裝", "泥作-打底", "防水-塗佈"]
    x = np.random.default_rng(0).normal(size=(30, 4))
    y = [classes[i % 3] for i in range(30)]
    clf = LogisticRegression(max_iter=200).fit(x, y)
    registry.split_path(name).write_text(
        json.dumps(
            {
                "name": name,
                "source": "report",
                "encoder": "siglip",
                "classes": classes,
                "pmsCatalogVersion": pms_review.catalog_version(),
                "train": [],
                "test": [],
                "labels": {},
            }
        ),
        encoding="utf-8",
    )
    registry.probe_path(name).write_bytes(pickle.dumps({"clf": clf, "split": name, "encoder": "siglip"}))


def test_export_bundle_writes_head_encoder_and_checked_metadata(pms_env):
    torch = pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    pytest.importorskip("safetensors")
    from types import SimpleNamespace

    from export_service_bundle import export_bundle

    _trained_version()
    out = registry.service_bundle_path("v9")
    fake = SimpleNamespace(visual=torch.nn.Linear(4, 4))  # 不下載真權重：只驗匯出流程與檔案契約
    export_bundle("v9", out, encoder=fake)

    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["modelVersion"] == "v9" and meta["encoderKey"] == "siglip"
    assert meta["catalogVersion"] == pms_review.catalog_version()
    assert sorted(meta["classes"]) == sorted(["油漆-塗裝", "泥作-打底", "防水-塗佈"])
    head = np.load(out / "classifier.npz")
    assert head["coef"].shape == (3, 4)
    assert (out / "encoder.safetensors").stat().st_mode & 0o044  # 容器內非 root 使用者讀得到
    assert not out.with_name("v9.building").exists()
    with pytest.raises(FileExistsError):  # 已匯出的包不覆蓋
        export_bundle("v9", out, encoder=fake)


def test_export_refuses_catalog_drift(pms_env):
    pytest.importorskip("open_clip")
    from export_service_bundle import export_bundle

    _trained_version()
    sp = registry.load_split("v9")
    sp["pmsCatalogVersion"] = "old"
    registry.split_path("v9").write_text(json.dumps(sp), encoding="utf-8")
    with pytest.raises(ValueError, match="分類表"):
        export_bundle("v9", registry.service_bundle_path("v9"))


def test_daily_export_moves_the_service_pointer_only_after_success(pms_env, monkeypatch):
    import daily
    import export_service_bundle

    def fake_export(version, out):
        out.mkdir(parents=True)
        (out / "metadata.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(export_service_bundle, "export_bundle", fake_export)
    (paths.MODELS / "service").mkdir(parents=True, exist_ok=True)
    assert registry.service_current() is None
    daily.export_service_bundle("v9")
    assert registry.service_current() == "v9"

    monkeypatch.setattr(
        export_service_bundle, "export_bundle", lambda v, o: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError):
        daily.export_service_bundle("v10")
    assert registry.service_current() == "v9"  # 匯出失敗不能把服務指到半成品
