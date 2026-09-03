"""Field Reports × Grounding DINO demo 操作台。

    uv run streamlit run src/fr_app.py

展示「AI 初標 → 人看 → 人審」的完整迴圈（ROADMAP 執行序 #1 的煙霧測試配套）：
- ① Run 總覽：詞彙命中統計、每樣態框數/均分、速度
- ② 照片牆：AI_GUESS 框即時畫在照片上，配原描述對照
- ③ 人審打分：對單框 mark correct / wrong / ambiguous——HUMAN_REFINED 的種子，
       結果寫進 data/field_reports/derived/gdino/verdicts.csv（append-only，
       與 review.csv 同精神：人手打的不可重算，不進 derived 洗牌範圍）
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import statistics
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import FR_PHOTOS, ROOT  # noqa: E402

GDINO_ROOT = ROOT / "data" / "field_reports" / "derived" / "gdino"
VERDICTS = GDINO_ROOT / "verdicts.csv"

# 樣態 → 顏色（固定色票，照片牆與人審同色）
PATTERN_COLORS: dict[str, str] = {
    "crack": "#e11d48",
    "gap_finish": "#f97316",
    "dirt_residue": "#a16207",
    "damage": "#dc2626",
    "scratch": "#db2777",
    "rust": "#b45309",
    "water": "#2563eb",
    "paint": "#7c3aed",
    "protection": "#059669",
    "custom": "#22d3ee",
}

# 描述關鍵字 → 樣態（tab ③ 交叉分析用：人寫描述提過 vs AI 真的框到）
DESC_PATTERNS: dict[str, str] = {
    "water": r"滲|漏|水痕|積水|潮",
    "dirt_residue": r"髒|汙|污|垢|垃圾|蜘蛛網|落葉|灰|殘留",
    "damage": r"破|脫落|掉|缺角|缺損|斷|鬆|壞",
    "gap_finish": r"縫|隙|收邊|收尾|填縫|勾縫|孔",
    "protection": r"保護",
    "cleanliness": r"清潔|清理|清洗",
}

st.set_page_config(page_title="Eagle Vision · GDINO 標註 demo", layout="wide", page_icon="🦅")
st.title("Grounding DINO anno demo")
st.caption("樂氧森 2026-08-28 · 469 張 · AI_GUESS 框僅為候選，人審後才是訓練級")


# ── 資料載入（st.cache_resource 抓模型以外的東西；run 清單用 cache_data）────
@st.cache_data
def list_runs() -> list[dict]:
    runs = []
    for d in sorted(GDINO_ROOT.glob("runs/*"), reverse=True):
        s = d / "summary.json"
        if s.exists():
            runs.append(json.loads(s.read_text()))
    return runs


@st.cache_data
def load_ann(run_id: str) -> list[dict]:
    out = []
    for p in sorted((GDINO_ROOT / "runs" / run_id / "annotations").glob("*.json")):
        out.append(json.loads(p.read_text()))
    return out


def load_verdicts() -> dict[str, str]:
    """verdicts.csv → {(runId, photoId, boxIdx): verdict}。"""
    if not VERDICTS.exists():
        return {}
    with open(VERDICTS, encoding="utf-8") as f:
        return {
            f"{r['runId']}|{r['photoId']}|{r['boxIdx']}": r["verdict"]
            for r in csv.DictReader(f)
        }


def append_verdict(run_id: str, photo_id: str, box_idx: int, verdict: str, note: str = "") -> None:
    VERDICTS.parent.mkdir(parents=True, exist_ok=True)
    new = not VERDICTS.exists()
    with open(VERDICTS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["runId", "photoId", "boxIdx", "verdict", "note", "at"])
        from datetime import datetime

        w.writerow([run_id, photo_id, box_idx, verdict, note, datetime.now().isoformat(timespec="seconds")])


# ── 畫框：PIL 在原圖上畫 AI_GUESS -------------------------------------------
def draw_boxes(image, boxes: list[dict], max_side: int = 1400):
    from PIL import ImageDraw

    im = image.copy()
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    dr = ImageDraw.Draw(im)
    for b in boxes:
        x1, y1, x2, y2 = [int(v * scale) for v in b["box"]]
        color = PATTERN_COLORS.get(b["pattern"], "#0ea5e9")
        dr.rectangle([x1, y1, x2, y2], outline=color, width=4)
        dr.text((x1 + 4, y1 + 4), f"{b['pattern']} {b['score']}", fill=color)
    return im


# ── ① Run 總覽 ─────────────────────────────────────────────────────────
runs = list_runs()
if not runs:
    st.warning("還沒有任何 run——先跑 `uv run --extra gdino src/fr_gdino.py --all`")
    st.stop()

run_ids = [f"{r['runId']} · {r['photos']}張 · {r['device']}" for r in runs]
pick = st.sidebar.selectbox("GDINO run", run_ids, index=0)
run = runs[run_ids.index(pick)]
run_id = run["runId"]
anns = load_ann(run_id)
verdicts = load_verdicts()

with st.sidebar:
    st.divider()
    st.markdown(f"**{run['modelId'].split('/')[-1]}** · {run['device']} · "
                f"{run['msPerPhoto']}ms/張")
    st.caption("box_thr {:.2f} · text_thr {:.2f}".format(run["boxThreshold"], run["textThreshold"]))
    pat_filter = st.multiselect(
        "只看含這些樣態的照片",
        list(run["vocabulary"].keys()),
        default=[],
    )
    only_hits = st.toggle("只看有框的照片", value=True)
    st.divider()
    n_v = sum(1 for k in verdicts if k.startswith(run_id))
    st.caption(f"本 run 已人審 {n_v} 框")

tab1, tab2, tab3, tab4 = st.tabs(["① 樣態統計", "② 照片牆 · 人審打分", "③ 煙霧測試分析", "④ 動態掃描 demo"])

with tab1:
    st.subheader("樣態可分性初判（AI_GUESS 層）")
    pp = run["perPattern"]
    rows = [
        {
            "樣態": p,
            "有框照片": v["photosWith"],
            "框數": v["boxes"],
            "均分": v["avgScore"],
            "覆蓋率": f"{v['photosWith'] / run['photos']:.1%}",
        }
        for p, v in pp.items()
    ]
    rows = sorted(rows, key=lambda r: -r["框數"])
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("注意：框數高 ≠ 樣態真的多——full-frame 假陽性是 GDINO 冷啟動的已知病，"
               "判斷工具素質要看照片牆裡框得準不準。")
    with st.expander("英文詞彙表"):
        st.json(run["vocabulary"])

with tab2:
    pool = anns
    if only_hits:
        pool = [a for a in pool if a["boxes"]]
    if pat_filter:
        wanted = set(pat_filter)
        pool = [a for a in pool if wanted & {b["pattern"] for b in a["boxes"]}]
    st.caption(f"符合條件：{len(pool)} / {len(anns)} 張")

    PAGE = 10
    page = st.number_input("頁", min_value=1, max_value=max(1, (len(pool) // PAGE) + 1), value=1)
    chunk = pool[(page - 1) * PAGE : (page - 1) * PAGE + PAGE]

    cols = st.columns(2)
    for i, a in enumerate(chunk):
        with cols[i % 2]:
            from PIL import Image

            img_path = FR_PHOTOS / a["file"]
            if not img_path.exists():
                st.error(f"照片不在：{a['file']}")
                continue
            with Image.open(img_path) as im:
                st.image(
                    draw_boxes(im.convert("RGB"), a["boxes"]),
                    caption=f"{a['photoDisplayId']} · {a['description'][:36] or '(無描述)'}",
                )
            if not a["boxes"]:
                st.caption("（AI 沒框到任何東西）")
            for bi, b in enumerate(a["boxes"]):
                key = f"{run_id}|{a['photoId']}|{bi}"
                cur = verdicts.get(key)
                color = PATTERN_COLORS.get(b["pattern"], "#0ea5e9")
                st.markdown(
                    f"<span style='color:{color}'>■</span> **{b['pattern']}** "
                    f"`{b['phrase']}` score={b['score']} box={b['box']}",
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                if c1.button("✓ 正確", key=f"ok-{key}", disabled=cur == "correct",
                             use_container_width=True):
                    append_verdict(run_id, a["photoId"], bi, "correct", b["phrase"])
                    st.rerun()
                if c2.button("✗ 錯框", key=f"no-{key}", disabled=cur == "wrong",
                             use_container_width=True):
                    append_verdict(run_id, a["photoId"], bi, "wrong", b["phrase"])
                    st.rerun()
                if c3.button("？ 模糊", key=f"amb-{key}", disabled=cur == "ambiguous",
                             use_container_width=True):
                    append_verdict(run_id, a["photoId"], bi, "ambiguous", b["phrase"])
                    st.rerun()
                if cur:
                    c4.markdown(f"已判：**{cur}**")

with tab3:
    st.subheader("煙霧測試交叉分析（本輪 run 的誠實數字）")

    # ── A. 覆蓋：人寫描述提過 vs AI 同張真的框到 ────────────────────────
    st.markdown("**A · 描述提及 ↔ GDINO 命中**（recall 的弱標籤版：描述提過該樣態，"
                "AI 同張有沒有框到同樣態）")
    cross_rows = []
    for pat, rx in DESC_PATTERNS.items():
        mentioned = hit = 0
        for a in anns:
            d = a["description"] or ""
            if re.search(rx, d):
                mentioned += 1
                if pat in {b["pattern"] for b in a["boxes"]}:
                    hit += 1
        cross_rows.append({
            "樣態": pat,
            "描述提及": mentioned,
            "AI 同張框到": hit,
            "命中率": f"{hit / mentioned:.0%}" if mentioned else "—",
        })
    cross_rows = sorted(cross_rows, key=lambda r: -(r["描述提及"]))
    st.dataframe(cross_rows, hide_index=True, width="stretch")
    st.caption("命中率低不是結案——描述是「人視角」的缺失，照片裡不一定看得到；"
               "但 0% 的樣態代表詞彙表或模型對該樣態無感，值得人工翻幾張確認。")

    # ── B. 框面積病：滿框假陽性與框大小分布 ────────────────────────────
    st.markdown("**B · 框面積分布**（滿框 = 假陽性的典型病徵；本輪詞彙已修訂，驗證修掉了沒）")
    area_rows = []
    full_frame = 0
    total_boxes = 0
    for pat in sorted({b["pattern"] for a in anns for b in a["boxes"]}):
        areas = []
        for a in anns:
            for b in a["boxes"]:
                if b["pattern"] == pat:
                    x1, y1, x2, y2 = b["box"]
                    areas.append((x2 - x1) * (y2 - y1) / (a["width"] * a["height"]))
                    total_boxes += 1
                    if areas[-1] > 0.8:
                        full_frame += 1
        area_rows.append({
            "樣態": pat,
            "框數": len(areas),
            "面積中位": f"{statistics.median(areas):.0%}",
            "滿框數": sum(1 for x in areas if x > 0.8),
        })
    st.dataframe(area_rows, hide_index=True, width="stretch")
    st.caption(f"滿框（>80% 畫面）合計：{full_frame}/{total_boxes}")

    # ── C. 分數貼門檻：所有框的 score 分布 ─────────────────────────────
    st.markdown("**C · 分數貼門檻程度**（全框都擠在門檻上方 = 門檻是主要決定者，"
                "降門檻的邊際要人審買單）")
    scores = [b["score"] for a in anns for b in a["boxes"]]
    if scores:
        thr = run["boxThreshold"]
        col1, col2, col3 = st.columns(3)
        col1.metric("框數", len(scores))
        col2.metric("均分", f"{statistics.mean(scores):.3f}")
        col3.metric("離門檻", f"+{statistics.mean(scores) - thr:.3f}")
        st.caption(f"門檻 {thr}；若降到 {thr - 0.05:.2f} 預期框量會明顯上升——"
                   f"降門檻前先看 wrong 率撐不撐得住（tab ② 打分）。")

    # ── D. 人審現況：verdicts 比例 ─────────────────────────────────────
    st.markdown("**D · 人審現況**（你在 tab ② 打的分，工具定案的最後一關）")
    v_counts = {"correct": 0, "wrong": 0, "ambiguous": 0}
    for k, v in verdicts.items():
        if k.startswith(run_id) and v in v_counts:
            v_counts[v] += 1
    done = sum(v_counts.values())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("已審", f"{done}/{total_boxes}")
    col2.metric("✓ correct", v_counts["correct"])
    col3.metric("✗ wrong", v_counts["wrong"])
    col4.metric("？ ambiguous", v_counts["ambiguous"])
    if done >= 30:
        prec = v_counts["correct"] / max(v_counts["correct"] + v_counts["wrong"], 1)
        st.success(f"目前 precision ≈ **{prec:.0%}**（correct/(correct+wrong)，n={done}）")
    else:
        st.info("審滿 30 框後這裡會出現 precision 估計——這就是初標工具定案的數字。")


# ═══════════════════════════════════════════════════════════════════════
# ④ 動態掃描 demo：機器「正在找目標」的感覺
#
# 動態感的四個要素：
#   1. 掃描線由上往下掃（CSS translateY 動畫，帶 glow）
#   2. 每個框按 y 座標排序出場——animation-delay = (y1/圖高)×掃描時間
#   3. 描邊動畫 stroke-dasharray=框周長 → dashoffset 動到 0，框像被「畫」出來
#   4. label chip 在框描完後淡入，分數 count-up
# ═══════════════════════════════════════════════════════════════════════
ANIM_CSS = """
<style>
@keyframes scanline { from { transform: translateY(0); opacity: 1; }
                      to   { transform: translateY(var(--scan-h)); opacity: .15; } }
@keyframes drawbox  { from { stroke-dashoffset: var(--perim); } to { stroke-dashoffset: 0; } }
@keyframes fadein   { from { opacity: 0; } to { opacity: 1; } }
.scanwrap { position: relative; overflow: hidden; border-radius: 8px;
            --scan-h: calc(100% - 3px); }
.scanline { position: absolute; left: 0; right: 0; top: 0; height: 3px; z-index: 2;
            background: linear-gradient(90deg, transparent, #22d3ee, transparent);
            box-shadow: 0 0 12px 2px #22d3eeaa; animation: scanline var(--scan-t) linear forwards; }
.overlay  { position: absolute; inset: 0; width: 100%; height: 100%; }
.gbox { fill: none; stroke-width: 6; stroke-linejoin: round;
        stroke-dasharray: var(--perim); stroke-dashoffset: var(--perim);  /* 延遲前隱形 */
        animation: drawbox .45s ease-out var(--delay) forwards;
        filter: drop-shadow(0 0 1px #000) drop-shadow(0 0 3px currentColor); }  /* 黑 halo：同色系照片上仍可見 */
.chip { opacity: 0; animation: fadein .3s ease-out var(--delay) forwards; }
</style>
"""


def scan_demo_html(img_b64: str, w: int, h: int, boxes: list[dict], scan_s: float = 2.4) -> str:
    """把照片＋AI框包成一次性的掃描動畫 HTML。

    全部元素（框、chip）都在同一個 SVG viewBox 裡，跟著 <img width:100%> 等比縮放，
    不受 component iframe 實寬影響——第一版用 img width=原圖像素 + 絕對像素 chip，
    在窄 iframe 裡右半全被裁掉、chip 錯位，不要走回頭路。
    """
    parts = [f'<div class="scanwrap" style="--scan-t:{scan_s}s">']
    parts.append(f'<img src="data:image/webp;base64,{img_b64}" style="display:block;width:100%">')
    parts.append(f'<div class="scanline"></div>')
    parts.append(f'<svg viewBox="0 0 {w} {h}" class="overlay">')
    for b in sorted(boxes, key=lambda b: b["box"][1]):  # 依 y 出場
        x1, y1, x2, y2 = b["box"]
        color = PATTERN_COLORS.get(b["pattern"], "#22d3ee")
        perim = 2 * ((x2 - x1) + (y2 - y1))
        delay = (y1 / max(h, 1)) * (scan_s * 0.8)  # 掃描線掃到那排的時刻
        label = f"{b['phrase']} {b['score']:.2f}"
        fs = 26                                   # viewBox 字號（隨圖寬等比縮放）
        tw = len(label) * fs * 0.58                # 粗估文字寬
        parts.append(
            f'<rect class="gbox" x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
            f'stroke="{color}" style="--perim:{perim};--delay:{delay:.2f}s"/>'
        )
        # chip 畫在框內左上；框太扁放不下就放框上方
        cy = y1 + 8 if (y2 - y1) > fs * 1.6 else y1 - fs * 1.4
        cy = max(cy, fs * 1.2)
        parts.append(
            f'<g class="chip" style="--delay:{delay + 0.45:.2f}s">'
            f'<rect x="{x1 + 8}" y="{cy - fs}" width="{tw}" height="{fs * 1.25}" rx="8" fill="{color}" opacity="0.92"/>'
            f'<text x="{x1 + 16}" y="{cy - fs * 0.2}" font-size="{fs}" font-weight="600" '
            f'font-family="ui-monospace,monospace" fill="#fff">{b["phrase"]} '
            f'<tspan class="cnt" data-target="{b["score"]:.2f}">0.00</tspan></text></g>'
        )
    parts.append("</svg>")
    # 分數 count-up：chip 淡入的同時，分數從 0.00 滾到實際值（rAF，300ms）
    parts.append(
        "<script>document.querySelectorAll('.cnt').forEach(el=>{const g=el.closest('g.chip');"
        "const d=parseFloat(g.style.getPropertyValue('--delay'))*1000;const t=+el.dataset.target;"
        "setTimeout(()=>{const t0=performance.now();const step=n=>{const p=Math.min(n/300,1);"
        "el.textContent=(t*p).toFixed(2);if(p<1)requestAnimationFrame(step)};requestAnimationFrame(step)},d)});"
        "</script>"
    )
    parts.append("</div>")
    return ANIM_CSS + "".join(parts)


with tab4:
    st.subheader("動態掃描 demo — 機器正在找目標的感覺")
    st.caption("框按 y 座標跟著掃描線出場、SVG 描邊動畫、分數即時浮出。"
               "兩種模式：重播既有 run 的框（左），或現場跑一張自訂詞彙（右，~1s/張）。")

    all_anns = load_ann(run_id)
    boxed = [a for a in all_anns if a["boxes"]]
    cand = st.selectbox(
        "挑一張有框的照片",
        [f"{a['photoDisplayId']} · {(a['description'] or '')[:28]}" for a in boxed],
    )
    cL, cR = st.columns(2)
    with cL:
        st.markdown("**重播 run 裡的框**")
        # 播放狀態進 session_state：Streamlit rerun 後按鈕會回 False，
        # 不存的話動畫會被任何一次互動洗掉（demo 現場致命）。
        if st.button("▶ 掃描播放", type="primary") or st.session_state.get("fr_play"):
            st.session_state["fr_play"] = cand
            from PIL import Image

            a = next(x for x in boxed if f"{x['photoDisplayId']} · {(x['description'] or '')[:28]}" == cand)
            with Image.open(FR_PHOTOS / a["file"]) as im:
                buf = io.BytesIO()
                im.convert("RGB").save(buf, "WEBP", quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"> {a['description'] or '(無描述)'}", unsafe_allow_html=True)
            components.html(scan_demo_html(b64, a["width"], a["height"], a["boxes"]), height=640)
            if st.button("■ 停止", help="清掉播放狀態"):
                st.session_state["fr_play"] = None
                st.rerun()

    with cR:
        st.markdown("**現場跑一張（自訂詞彙）**")
        with st.form("live_run"):
            file_pick = st.selectbox("照片", [a["file"] for a in all_anns[:200]])
            prompt_in = st.text_input(
                "英文詞彙（' . ' 分隔）", "dirt . stain . gap between tiles . puddle",
            )
            live = st.form_submit_button("跑 GDINO → 動畫", type="primary")
        if live:
            from PIL import Image

            from fr_gdino import GdinoRunner

            with st.spinner("GDINO 推理中…"):
                runner = st.cache_resource(GdinoRunner)()
                with Image.open(FR_PHOTOS / file_pick) as im:
                    img = im.convert("RGB")
                    boxes = runner.infer_raw(img, prompt_in, 0.35, 0.25)
            st.markdown(f"框到 **{len(boxes)}** 個目標")
            if boxes:
                buf = io.BytesIO()
                img.save(buf, "WEBP", quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode()
                components.html(scan_demo_html(b64, img.width, img.height, boxes), height=640)
