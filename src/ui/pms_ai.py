"""照片工種與進階複核共用的 AI 分類操作。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core import pms_exchange as exchange
from core import pms_review as review
from core import pms_store as store
from core import pms_vision as vision

RELATIONS = {
    "agrees": "標題與圖片一致",
    "conflicts": "標題與圖片有衝突",
    "insufficient_context": "標題或圖片資訊不足",
}
DECISIONS = {
    "existing": "既有工種",
    "new_candidate": "新工種候選",
    "uncertain": "資訊不足",
    "not_construction": "非本輪工種語料",
}


def ai_panel(view: pd.DataFrame, key: str) -> dict | None:
    """只有使用者按下分類才送出請求；回傳結果供呼叫端更新佇列。"""
    with st.expander("AI 自動分類：照片＋日報標題", expanded=False):
        st.caption("選取施作項目照片後，AI 會看圖片、讀標題並回傳工種與理由。結果保存為待確認建議。")
        config = vision.configuration()
        if not config["keyConfigured"]:
            st.info("尚未設定 OPENAI_API_KEY。設定後即可直接辨識；也可繼續使用本機審閱包。")
        model = st.text_input(
            "OpenAI 視覺模型 ID",
            value=config["model"],
            placeholder="填入帳號可用且支援圖片與結構化輸出的模型",
            key=f"{key}_model",
        )
        subset = view.head(200)
        options = subset.fileId.tolist()
        lookup = {
            r.fileId: f"{r.reportDate} · {r.title or '無標題'} · {r.fileId[:8]}" for r in subset.itertuples()
        }
        selection_key = f"{key}_ids"
        if selection_key in st.session_state:
            kept = [f for f in st.session_state[selection_key] if f in lookup]
            if kept != st.session_state[selection_key]:
                st.session_state[selection_key] = kept
        ids = st.multiselect(
            f"本批辨識照片（最多 {vision.MAX_BATCH} 張）",
            options,
            default=options[:4],
            format_func=lambda fid: lookup[fid],
            key=selection_key,
        )
        force = st.checkbox("重新辨識已有的 AI 結果", key=f"{key}_force")
        st.caption("按下按鈕會將選取的照片、日報標題、查驗重點及分類表傳至 OpenAI；API 用量另行計費。")
        disabled = not config["keyConfigured"] or not model.strip() or not 1 <= len(ids) <= vision.MAX_BATCH
        if st.button("AI 看圖＋標題分類", disabled=disabled, key=f"{key}_run"):
            try:
                bar = st.progress(0.0, text="開始辨識選取照片")

                def progress(event):
                    bar.progress(
                        event["done"] / event["total"], text=f"已處理 {event['done']} / {event['total']} 張"
                    )

                result = vision.classify(ids, model=model, force=force, progress=progress)
                st.session_state[f"{key}_result"] = result
                return result
            except (ValueError, OSError, RuntimeError) as exc:
                st.error(str(exc))
        result = st.session_state.get(f"{key}_result")
        if result:
            st.write(
                f"上批：新增 {result['completed']} 張分類建議、沿用 {result['cached']} 張、"
                f"失敗 {len(result['failed'])} 張。"
            )
            if result["failed"]:
                st.dataframe(pd.DataFrame(result["failed"]), hide_index=True)
    return None


def suggestion_card(proposal: dict, reviewer: str, *, key: str, expected_revision: str) -> bool:
    """顯示建議及採用動作；採用前再次核對照片版本與人工裁決。"""
    if not proposal:
        return False
    st.info(f"AI 建議：{proposal['label'] or DECISIONS[proposal['decision']]} · 來源 {proposal['model']}")
    st.write(f"圖片證據：{proposal['visualEvidence']}")
    if proposal.get("titleEvidence"):
        st.write(f"標題參考：{proposal['titleEvidence']}")
        st.caption(RELATIONS[proposal["titleRelation"]])
    st.write(f"判斷理由：{proposal['reason']}")
    try:
        exchange.validate_proposal(proposal)
        fresh = True
    except (ValueError, OSError) as exc:
        fresh = False
        st.warning(f"此建議已不能直接採用：{exc}")
    if proposal["decision"] == "existing" and st.button(
        "確認採用此 AI 類別", key=key, disabled=not fresh or not reviewer.strip()
    ):
        try:
            review.decide(
                proposal["fileId"],
                "classified",
                reviewer=reviewer,
                label=proposal["label"],
                proposal_id=proposal["proposalId"],
                reason="看圖後確認 AI 建議",
                expected_revision=expected_revision,
            )
            return True
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    elif proposal["decision"] == "new_candidate":
        st.caption("新類建議需核對分類定義；請在照片工種頁提出候選。")
    return False


def latest_suggestions() -> dict[str, dict]:
    return {r["fileId"]: r for r in store.events("suggestion")}
