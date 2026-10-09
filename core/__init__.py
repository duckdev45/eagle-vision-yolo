# core/__init__.py — 服務層套件（QS / 合約 / 標籤規則 / 模型版本 / PMS 照片與裁決 / 分流）
#
# 匯入路徑由 editable 安裝負責（pyproject 的 hatch dev-mode-dirs 把 repo 根與 src/ 放上 sys.path），
# 這裡不再動 sys.path。路徑常數在 core/paths.py——單一 module object，
# tests 的 monkeypatch.setattr(paths, ...) 才會全域生效。
