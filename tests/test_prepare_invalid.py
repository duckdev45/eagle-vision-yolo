"""前處理遇到 PMS 上游的極小壞檔時，仍可完成其餘照片。"""

from random import Random

from PIL import Image


def test_prepare_skips_tiny_raw_file_without_stopping(tmp_path, monkeypatch):
    import prepare

    raw = tmp_path / "raw"
    images = tmp_path / "images"
    raw.mkdir()
    images.mkdir()
    bad = raw / "bad.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0test-jpeg-bytes")
    (raw / "bigbad.jpg").write_bytes(b"not-a-jpeg" * 150)
    rng = Random(0)
    Image.frombytes("RGB", (128, 128), rng.randbytes(128 * 128 * 3)).save(raw / "good.jpg", quality=95)
    monkeypatch.setitem(prepare.SRC_DIRS, "report", (raw, images))
    monkeypatch.setattr(prepare.paths, "ensure_dirs", lambda: None)
    monkeypatch.setattr(prepare, "clean_ids", lambda: set())

    messages = []
    result = prepare.run(log=messages.append)

    assert result == {"processed": 1, "skipped": 0, "failed": 0, "unmasked": 0, "invalid": 2}
    assert bad.read_bytes() == b"\xff\xd8\xff\xe0test-jpeg-bytes"
    assert (images / "good.jpg").exists()
    assert not (images / "bad.jpg").exists()
    assert not (images / "bigbad.jpg").exists()
    assert any("跳過壞檔 bad.jpg" in line for line in messages)
    assert any("跳過壞檔 bigbad.jpg" in line for line in messages)
