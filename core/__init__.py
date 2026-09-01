# core/__init__.py — 服務層套件（QS / 合約 / 標籤規則 / 複核計算）
#
# 路徑定義的實體在 src/paths.py（單一名字 "paths"，所有模組共用同一個 module
# object——tests 的 monkeypatch.setattr(paths, ...) 才會全域生效）。
# 這裡統一把 root 與 src 塞進 sys.path，之後 core 服務層與 src 的 `import paths`
# 拿到的都是同一個物件。
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_SRC = _os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
