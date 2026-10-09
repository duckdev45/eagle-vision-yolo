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


def _core_modules_imported(path: Path) -> set[str]:
    """`from core import x`、`from core.x import y`、`import core.x` → {"x", ...}。"""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module == "core":
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("core."):
            out.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            out.update(a.name.split(".")[1] for a in node.names if a.name.startswith("core."))
    return out


def test_workbench_view_is_a_leaf_of_core():
    """pms_review 是工作台檢視（組表給畫面），可以依賴 routing；反過來任何 core 模組 import 它
    就會回到 2026-10-09 以前的循環（routing ⇄ pms_review 要靠函式內延遲 import 撐著）。"""
    offenders = [
        p.name
        for p in (ROOT / "core").glob("*.py")
        if p.name != "pms_review.py" and "pms_review" in _core_modules_imported(p)
    ]
    assert not offenders, f"core 模組不得 import 工作台檢視 pms_review：{offenders}"


def test_human_verdicts_are_written_only_through_pms_decisions():
    """review.csv 的寫入（save_review）只准出現在 core/pms_decisions.py——收件匣、工作台、標框頁、
    CLI、未來標註平台都經 decide()，驗證（母體、確認者、類名）才不會有漏網的入口。"""
    offenders = [
        f"{p.relative_to(ROOT)}:{n}"
        for top in ("core", "src", "pipeline")
        for p in (ROOT / top).rglob("*.py")
        if p.name not in {"pms_decisions.py", "labeler.py"}
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"\bsave_review\(", line)
    ]
    assert not offenders, offenders
