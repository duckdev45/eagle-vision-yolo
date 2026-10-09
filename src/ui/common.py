"""操作台共用元件：CSS、分頁導航、小工具。

這些函式被多個分頁共用，放最上層而非某個分頁裡——分頁的程式碼會因為
「那批資料還沒有」而整段跳過，樣式跟著消失的話其他分頁的卡片會裸奔。
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import streamlit as st

_CSS = """<style>
.pc{height:114px;line-height:1.5;font-size:12px;overflow:hidden}
.pc6{height:132px;line-height:1.5;font-size:12px;overflow:hidden}
.pc div,.pc6 div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pc .m,.pc6 .m{color:#8b949e}
.pc .ok,.pc6 .ok{color:#3fb950}
.pc .no,.pc6 .no{color:#e5534b}
.bdg{display:inline-block;padding:0 6px;border-radius:3px;font-size:11px;
     font-weight:700;color:#fff;letter-spacing:.5px}
.bdg.tr{background:#2f9e44}.bdg.te{background:#c92a2a}
/* 複核卡片：來源標籤。名稱與值成對，一眼看得出誰說了什麼 */
.src{display:inline-block;background:#2a2a28;color:#9aa0a6;border-radius:3px;
     padding:0 5px;margin-right:2px;font-size:11px}
</style>"""


def inject_css() -> None:
    """CSS 注入。刻意包成函式、由 app.py 在 set_page_config 之後呼叫——
    在 import 時就執行 st.markdown 的話，set_page_config 就不是第一個 st
    指令，wide layout 會被 Streamlit 靜默忽略。放最上層而非某個分頁裡：
    分頁的程式碼會因為「那批資料還沒有」而整段跳過，樣式跟著消失的話
    其他分頁的卡片會裸奔。
    """
    st.markdown(_CSS, unsafe_allow_html=True)


def nav(names: list[str], key: str) -> list[bool]:
    """分頁選單。刻意不用 `st.tabs`：它的選取是純前端狀態，元件樹一變
    （例如標框畫布出現）就重置回第一頁——標一次框就被彈走一次。
    radio 的值存在 session_state，任何 rerun 都不會掉。
    """
    pick = st.radio(" ", names, horizontal=True, key=key, label_visibility="collapsed")
    return [pick == n for n in names]


def reviewer_input() -> str:
    """確認者：預設帶 .env 的 PMS_REVIEWER，不必每次手打（收件匣與工作台共用同一個 key）。"""
    return st.text_input(
        "確認者",
        value=os.getenv("PMS_REVIEWER", ""),
        key="pms_reviewer",
        placeholder="保存判斷時需要填寫（可在 .env 設 PMS_REVIEWER）",
    )


def run_step(fn, **kw):
    buf = io.StringIO()
    with st.spinner("執行中…"), redirect_stdout(buf):
        try:
            fn(log=lambda *a: print(*a), **kw)
        except Exception as e:  # 操作台不該因為一個步驟炸掉就整頁死掉
            print(f"錯誤：{e}")
    st.code(buf.getvalue() or "(無輸出)")
    st.cache_data.clear()


def txt(v) -> str:
    """欄位可能是 NaN（pandas 讀 CSV 的空值）——NaN 是 truthy，`v or '－'` 擋不住。"""
    return "－" if v is None or v != v else str(v)


def verdict(label: str, p, truth) -> str:
    if p is None or p != p:
        return f"<div class=m>{label}：－</div>"
    return f'<div class={"ok" if p == truth else "no"}">{label}：{p}</div>'


def gem_line(raw, norm, truth, conf_s: str = "") -> str:
    """Gemini 那一行：顯示它自己寫的字，但用正規化後的結果決定綠/紅。

    正規化（`景觀草皮鋪設` → `植栽-景觀`）是我們為了跟 cls 比對硬折過去的，
    看板上要看的是它到底講了什麼。折出來的結果不在訓練類別裡時補一個「類別外」
    ——那不是它答錯，是我們沒有那個類別可以接。
    """
    if raw is None or raw != raw or not str(raw).strip():
        return "<div class=m>Gemini判斷：－</div>"
    cls_ = "ok" if norm == truth else "no"
    tail = " ·類別外" if norm == "類別外" else ""
    return (
        f'<div class={cls_} title="{txt(raw)}">Gemini判斷：{txt(raw)}{conf_s}'
        f"<span class=m>{tail}</span></div>"
    )


def badge(part: str) -> str:
    """test 紅、其餘綠：紅色代表「模型沒背過，這些才算數」。"""
    return f'<span class="bdg {"te" if part == "test" else "tr"}">{part.upper()}</span>'
