"""操作台 UI 套件。

從 app.py 拆出的各分頁與共用元件。app.py 只剩進入點：
選頁面（PMS 各分頁／規範庫）→ 呼叫對應 page 模組。

模組依賴方向（禁止反向）：
    app.py → ui.pages.* → ui.{common,data,pipeline,review_ui,report_view} → src 各腳本

nav() 刻意不用 st.tabs：它的選取是純前端狀態，元件樹一變（例如標框畫布出現）
就重置回第一頁——標一次框就被彈走一次。radio 的值存在 session_state，
任何 rerun 都不會掉。
"""

from .common import badge, gem_line, inject_css, nav, run_step, txt, verdict
from .data import labeled, load_manifest
from .report_view import report_view

__all__ = [
    "badge",
    "gem_line",
    "inject_css",
    "labeled",
    "load_manifest",
    "nav",
    "report_view",
    "run_step",
    "txt",
    "verdict",
]
