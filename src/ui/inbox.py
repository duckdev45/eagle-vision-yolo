"""收件匣：每日分流後真正需要人看的照片，一次一張、一鍵裁決。

只做畫面；分流規則在 core/routing.py，寫入走 routing.resolve()（= 工作台同一條 decide）。
取代舊「③ 照片工種／④ 進階複核」的日常用途——那兩頁還在「進階」裡，畫框、缺失旗標、提新類仍去那邊。
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from core import pms_review as review
from core import routing

from .common import reviewer_input, run_step

_IDX = "inbox_idx"


def _summary(meta: dict, open_count: int) -> None:
    counts = meta.get("counts", {})
    audit = meta.get("auditPrecision") or {}
    a, b, c, d = st.columns(4)
    a.metric("待看", open_count)
    b.metric("自動確認", counts.get("auto", 0))
    c.metric(
        "隔離", counts.get("quarantine", 0), help="沒有原圖、沒有圖像特徵、日期不合理——不是人能裁的，要修資料"
    )
    prec = audit.get("precision")
    d.metric(
        "自動桶抽查準確率",
        "—" if prec is None else f"{prec:.1%}",
        help=f"抽查 {audit.get('sampled', 0)} 張、已有人看 {audit.get('reviewed', 0)} 張。"
        "這是自動確認唯一可信的準確率，抽查照片也在收件匣裡。",
    )
    when = meta.get("routedAt", "")
    try:
        when = datetime.fromisoformat(when).astimezone().strftime("%m-%d %H:%M")
    except ValueError:
        pass
    st.caption(
        f"最近分流 {when} · 模型 {meta.get('model', '?')} · 門檻 {meta.get('autoConfidence', routing.AUTO_CONFIDENCE)}"
        " · 自動確認不寫入人審，只決定誰需要看"
    )


def inbox() -> None:
    st.subheader("收件匣")
    if st.session_state.get("inbox_flash"):
        st.success(st.session_state.pop("inbox_flash"))
    items = routing.queue_items()
    _, meta = routing.latest()
    if not meta:
        st.info("還沒有分流結果。每日排程會自動產生；要現在算就按下面這顆。")
        if st.button("▶ 立即分流", type="primary"):
            run_step(routing.build)
            st.rerun()
        return
    _summary(meta, len(items))
    if items.empty:
        st.success("今天沒有需要人看的照片。")
        return

    reasons = [
        r for r in routing.REASONS[:-1] if items.reason.str.contains(r, regex=False).any()
    ]  # 抽查不給篩
    pick = st.segmented_control(
        "篩選", ["全部", *reasons], default="全部", key="inbox_filter", label_visibility="collapsed"
    )
    if pick and pick != "全部":
        items = items[items.reason.str.contains(pick, regex=False)].reset_index(drop=True)
    if items.empty:
        st.info("這個條件下已經看完了。")
        return
    reviewer = reviewer_input()
    idx = min(st.session_state.get(_IDX, 0), len(items) - 1)
    row = items.iloc[idx].to_dict()
    fid = str(row["fileId"])
    pool = review.load_pool().set_index("fileId")
    title = str(pool.title.get(fid, "")) if fid in pool.index else ""

    left, right = st.columns([3, 2])
    with left:
        try:
            st.image(review.photo_bytes(fid), width="stretch")
        except (ValueError, OSError) as exc:
            st.warning(f"無法讀取照片：{exc}")
    with right:
        st.markdown(f"**{idx + 1} / {len(items)}**　{row['reportDate']}")
        st.markdown(f"**標題**：{title or '（空白）'}")
        # 抽查照不告訴看的人「這是抽查」：知道是機器已經確認過的，人就容易照單全收，量出來的準確率會偏高
        shown = "例行確認" if row["bucket"] == "audit" else row["reason"]
        st.markdown(f"**為什麼要看**：{shown}")
        conf = row.get("modelConfidence")
        conf_s = "" if conf != conf or conf is None else f"（{float(conf):.0%}）"
        st.markdown(f"**規則**：{row['ruleClass'] or '—'}　**模型**：{row['modelClass'] or '—'}{conf_s}")

        def save(label: str = "", action: str = "classified", reason: str = "") -> None:
            try:
                routing.resolve(fid, reviewer=reviewer, label=label, action=action, reason=reason)
            except ValueError as exc:
                st.error(str(exc))
                return
            st.session_state["inbox_flash"] = f"已存：{label or action}"
            st.cache_data.clear()
            st.rerun()

        choices = list(dict.fromkeys(c for c in (row["ruleClass"], row["modelClass"]) if c))
        for i, label in enumerate(choices):
            source = "規則" if label == row["ruleClass"] else "模型"
            if label == row["ruleClass"] == row["modelClass"]:
                source = "規則＝模型"
            if st.button(
                f"✓ {label}（{source}）", key=f"inbox_pick_{i}", type="primary" if i == 0 else "secondary"
            ):
                save(label)
        other = st.selectbox("都不是 → 選正確類別", ["", *sorted(review.catalog())], key=f"inbox_other_{fid}")
        if other and st.button(f"✓ 存為 {other}", key="inbox_other_save"):
            save(other)
        why = st.text_input("看不出來／不該分類的原因", key=f"inbox_why_{fid}")
        c1, c2 = st.columns(2)
        if c1.button("暫緩（看不出來）", key="inbox_uncertain", disabled=not why.strip()):
            save(action="uncertain", reason=why)
        if c2.button("排除（不該分類）", key="inbox_exclude", disabled=not why.strip()):
            save(action="excluded", reason=why)
        n1, n2 = st.columns(2)
        if n1.button("← 上一張", key="inbox_prev", disabled=idx == 0):
            st.session_state[_IDX] = idx - 1
            st.rerun()
        if n2.button("略過 →", key="inbox_next", disabled=idx >= len(items) - 1):
            st.session_state[_IDX] = idx + 1
            st.rerun()
        st.caption("畫證據框、標缺失旗標、提新工種：到「進階 → 照片工種」。")


def status_panel() -> None:
    """總覽頁頂端：每日排程有沒有在跑、最近有沒有自動換模型。"""
    runs = routing.recent_runs()
    st.subheader("每日排程")
    if not runs["daily"]:
        st.info("尚無每日排程紀錄（make daily 或 hermes cron 跑過一次後出現）。")
    else:
        rows = []
        for r in runs["daily"]:
            counts = (r.get("route") or {}).get("counts", {})
            failed = [k for k, v in r.get("steps", {}).items() if v != "ok"]
            rows.append(
                {
                    "開始": r.get("startedAt", "")[:16].replace("T", " "),
                    "待看": counts.get("queue", 0) + counts.get("audit", 0),
                    "自動確認": counts.get("auto", 0),
                    "重訓": (r.get("retrain") or {}).get("why", ""),
                    "切換": {True: "✅", False: "⏸"}.get((r.get("promotion") or {}).get("promoted"), ""),
                    "失敗步驟": "、".join(failed),
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")
    if runs["promotions"]:
        with st.expander("自動切換紀錄（公平考卷）"):
            for p in runs["promotions"]:
                exam = p.get("exam") or {}
                cand, base = exam.get("candidateScore") or {}, exam.get("baselineScore") or {}
                head = f"{p.get('at', '')[:10]}　{p.get('baseline')} → {p.get('candidate')}　"
                head += "✅ 已切換" if p.get("promoted") else "⏸ 未切換"
                st.markdown(f"**{head}**")
                if cand and cand.get("top1") is not None:
                    st.caption(
                        f"考卷 {exam.get('examSize')} 張 · top1 {base['top1']:.3f} → {cand['top1']:.3f}"
                        f" · macroF1 {base['macroF1']:.3f} → {cand['macroF1']:.3f}"
                    )
                for why in p.get("reasons") or []:
                    st.caption(f"・{why}")
