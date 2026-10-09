"""架構邊界的守門：資料夾分層只是約定，沒有測試擋著就會慢慢爛掉。"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_MODULES = {p.stem for p in (ROOT / "src").glob("*.py")}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):  # 含函式內的延遲 import
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module.split(".")[0])
    return out


def test_core_does_not_reach_up_into_scripts():
    leaks = {
        f"{p.relative_to(ROOT)} → {m}"
        for p in (ROOT / "core").glob("*.py")
        for m in _imported_modules(p) & SRC_MODULES
    }
    # core 是服務層，只能往下依賴（零例外）。腳本需要的共用邏輯搬進 core，
    # 編排端要的副作用（例如匯出服務包）由呼叫端注入，見 core/promotion.promote(export=...)。
    assert not leaks, f"core 反向依賴 src 腳本：{sorted(leaks)}"


def test_no_sys_path_hacks():
    """匯入路徑由 editable 安裝負責（pyproject [tool.hatch.build]）；檔案自己插 sys.path 會跟它打架。"""
    offenders = [
        f"{p.relative_to(ROOT)}:{n}"
        for top in ("core", "src", "pipeline", "tests", "scripts-analysis")
        for p in (ROOT / top).rglob("*.py")
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"sys\.path\s*(\.insert|\.append|\[)", line)
    ]
    assert not offenders, offenders


def test_model_artifact_names_are_spelled_only_in_registry():
    """探針、split、特徵檔、CURRENT 的檔名拼法只准出現在 core/model_registry.py（測試自己造檔除外）。"""
    pattern = re.compile(r'probe-\{|SPLITS / f"\{|SPLITS / "CURRENT"|FEATURES / f"\{')
    offenders = [
        f"{p.relative_to(ROOT)}:{n}"
        for top in ("core", "src", "pipeline", "scripts-analysis")
        for p in (ROOT / top).rglob("*.py")
        if p.name != "model_registry.py"
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, f"自己拼模型檔名了，改用 core/model_registry.py：{offenders}"
