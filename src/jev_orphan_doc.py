"""孤兒照片（QS 有標準、labels.yaml 沒類）→ 看圖複核活文件。

純讀標題就發現「扶手」這桶根本不是一個工種：93 張裡真正的「扶手本體施作」只有 13 張，
其他 56 張拍的是修缺、油漆、拆除 —— 跟你 v9/v12 處理「缺失改善」是同一個結構問題。

所以這份文件不是問「扶手該不該成類」，而是逐格問：**這張照片的主體到底是什麼工項？**
分組依「動作」而非「名詞」，因為動作才決定計價與班組。

照片內嵌、勾選存 localStorage、匯出 JSON。
"""

from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import paths

IMG = Path("data/legacy/derived/images")
OUT = Path("reports/2026-09-19-jev/ORPHAN-REVIEW.html")

# ---- 分組：依「動作」不依名詞 ----------------------------------------------
GROUPS: dict[str, list[str]] = {
    "扶手-缺失改善": ["缺失改善", "缺改", "高度缺"],
    "扶手-油漆收尾": ["油漆", "染色", "補土", "打磨"],
    "扶手-本體施作": ["施作", "安裝", "裝設"],
    "扶手-拆除": ["拆除"],
}

HINT = {
    "扶手-缺失改善": (
        "這些標題寫「扶手」但動作是<b>修缺</b>。照片主體可能是：木扶手刮傷/缺口、"
        "高度不符、接合鬆動。要確認的是 —— 畫面強調的是「缺失本身」（油漆類）"
        "還是「修補後的成品」（扶手類）？v9/v12 的原則：缺失照片歸缺失，不歸工項。"
    ),
    "扶手-油漆收尾": (
        "動作是<b>油漆/補土/染色</b>。工地現場這類照片常拍的是「色差、流掛、補土痕」，"
        "扶手只是入鏡。若主體是漆面 → 歸 油漆-批土塗裝；若主體是扶手木料本身 → 留在扶手。"
    ),
    "扶手-本體施作": (
        "這組才是真正的「扶手工程」。要確認材質：<b>櫸木</b>（木作班）vs "
        "<b>店舖金屬</b>（鐵工班）—— 兩者計價與 QS 標準不同。"
        "QS 樹有獨立「扶手工程標準」一支。"
    ),
    "扶手-拆除": "臨時扶手拆除，屬雜項/清運。確認畫面是否為「拆除中」而非成品。",
}


def assign_group(title: str) -> str:
    if "扶手" not in title:
        return ""
    for g, keys in GROUPS.items():
        if any(k in title for k in keys):
            return g
    return "扶手-其他"


def other_bucket(leg: pd.DataFrame) -> pd.DataFrame:
    """格柵/捲門/刮泥/塑鋼 —— 各自張數不足 12，但 QS 有獨立標準，值得逐格看。"""
    out = []
    for name, pat in [
        ("格柵-冷氣", "冷氣格柵"),
        ("格柵-天花板", "天花板格柵"),
        ("格柵-防墜機房", "格柵"),
        ("捲門", "捲門"),
        ("刮泥墊", "刮泥"),
        ("塑鋼門", "塑鋼"),
    ]:
        h = leg[leg.title.str.contains(pat, na=False)]
        out.append(h.assign(candidate=name))
    o = pd.concat(out)
    # 格柵-防墜機房 要扣掉冷氣/天花板那兩組
    mask = (o.candidate == "格柵-防墜機房") & o.title.str.contains("冷氣|天花板", na=False)
    return o[~mask].sort_values(["candidate", "title"])


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def card(row: pd.DataFrame, idx: int, total: int) -> str:
    p = IMG / f"{row.fileId}.jpg"
    if not p.exists():
        return ""
    img = f"data:image/jpeg;base64,{b64(p)}"
    title = html.escape(str(row.title))
    return f"""
<div class="card" data-key="{row.fileId}" data-cand="{row.candidate}">
  <div class="imgwrap"><img loading="lazy" src="{img}" alt="{title}"></div>
  <div class="cap">{title}</div>
  <div class="meta">{row.candidate} · {row.constrName} · {row.reportDate} · {idx + 1}/{total}</div>
  <div class="opts">
    <label><input type="radio" name="v_{row.fileId}" value="keep"> 歸此類</label>
    <label><input type="radio" name="v_{row.fileId}" value="paint"> 其實是油漆</label>
    <label><input type="radio" name="v_{row.fileId}" value="defect"> 其實是缺失照</label>
    <label><input type="radio" name="v_{row.fileId}" value="drop"> 不是施工照</label>
  </div>
  <input class="note" placeholder="看圖備註…">
</div>"""


def main() -> int:
    leg = pd.read_csv(paths.LEGACY_MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
    leg = leg[leg.title.str.contains("扶手|格柵|捲門|刮泥|塑鋼", na=False)].copy()
    leg["candidate"] = leg.title.map(assign_group)

    # 扶手分組 + 其他六小桶合併標註
    others = other_bucket(leg[leg.candidate == ""])
    hand = leg[leg.candidate != ""].sort_values(["candidate", "title"])
    allr = pd.concat([hand, others], ignore_index=True)

    n_photo = len(allr)
    n_title = allr.title.nunique()
    print(f"照片 {n_photo} 張 / {n_title} 種標題 / {allr.candidate.nunique()} 個候選桶")

    sections = []
    order = [
        *GROUPS,
        "格柵-冷氣",
        "格柵-天花板",
        "格柵-防墜機房",
        "捲門",
        "刮泥墊",
        "塑鋼門",
    ]
    for cand in order:
        sub = allr[allr.candidate == cand]
        if sub.empty:
            continue
        hint = HINT.get(cand, "")
        cards = "".join(card(r, i, len(sub)) for i, (_, r) in enumerate(sub.iterrows()))
        sections.append(
            f"""
<section data-cand="{cand}">
  <h2>{cand} <span class="cnt">{len(sub)} 張 / {sub.title.nunique()} 種</span></h2>
  <div class="hint">{hint}</div>
  <div class="grid">{cards}</div>
</section>"""
        )

    OUT.write_text(
        DOC.replace("{{SECTIONS}}", "".join(sections)).replace("{{N}}", str(n_photo)),
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


DOC = """<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>孤兒照片複核 — Eagle Vision</title>
<style>
:root { color-scheme: light dark; }
body { font-family: var(--font, -apple-system, "PingFang TC", sans-serif);
       color: var(--foreground, #222); background: var(--card, #fff);
       margin: 0; padding: 16px; }
header { position: sticky; top: 0; background: var(--card, #fff);
         padding: 8px 0; border-bottom: 1px solid var(--border, #ddd); z-index: 9; }
h1 { font-size: 18px; margin: 4px 0; }
h2 { font-size: 16px; margin: 24px 0 4px; }
.cnt { font-weight: 400; opacity: .65; font-size: 13px; }
.hint { font-size: 13px; opacity: .8; margin: 4px 0 12px; max-width: 70ch;
        line-height: 1.6; border-left: 3px solid var(--accent, #08c);
        padding-left: 10px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
        gap: 12px; }
.card { border: 1px solid var(--border, #ccc); border-radius: 8px;
        padding: 8px; background: var(--card, #fff); }
.imgwrap { aspect-ratio: 4/3; overflow: hidden; }
.imgwrap img { width: 100%; height: 100%; object-fit: cover; cursor: zoom-in;
               transition: transform .2s; }
.imgwrap img:hover { transform: scale(1.04); }
.cap { font-size: 13px; margin: 6px 0 2px; line-height: 1.4; }
.meta { font-size: 11px; opacity: .55; }
.opts { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 8px;
        font-size: 12px; margin: 8px 0 4px; }
.opts label { cursor: pointer; }
.note { width: 100%; box-sizing: border-box; font-size: 12px;
        padding: 4px; border: 1px solid var(--border, #ccc);
        background: transparent; color: inherit; }
.card.done { outline: 2px solid var(--accent, #08c); }
button { font: inherit; padding: 6px 14px; border-radius: 6px; cursor: pointer;
         border: 1px solid var(--border, #ccc);
         background: var(--accent, #08c); color: #fff; }
button.ghost { background: transparent; color: var(--foreground, #222); }
#lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.88);
            display: none; align-items: center; justify-content: center; z-index: 99; }
#lightbox img { max-width: 92vw; max-height: 88vh; }
#lightbox.show { display: flex; }
footer { margin: 32px 0 8px; font-size: 12px; opacity: .55; }
</style></head><body>
<header>
  <h1>孤兒照片複核 — QS 有標準、labels.yaml 沒類</h1>
  <div style="font-size:12px;opacity:.7">
    {{N}} 張 · 看圖逐格判定 · 勾選存本機 · 匯出 JSON 後再改 labels.yaml（v11 紀律：看圖確認才收編）
  </div>
  <div style="margin:8px 0">
    <button id="export">匯出決定</button>
    <button class="ghost" id="reset">清空</button>
    <span id="progress" style="font-size:12px;opacity:.7;margin-left:8px"></span>
  </div>
</header>
{{SECTIONS}}
<div id="lightbox"><img id="lbimg"></div>
<footer>
  分組依「動作」不依名詞：修缺 / 油漆收尾 / 本體施作 / 拆除 是四件不同的事。<br>
  產生器：src/jev_review_doc.py 的姊妹作（同分支）。
</footer>
<script>
const KEY = 'orphan-review-v1';
document.querySelectorAll('.imgwrap img').forEach(img => {
  img.addEventListener('click', () => {
    document.getElementById('lbimg').src = img.src;
    document.getElementById('lightbox').classList.add('show');
  });
});
document.getElementById('lightbox').addEventListener('click', e =>
  e.currentTarget.classList.remove('show'));

function load() {
  try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { return {}; }
}
function save(d) { localStorage.setItem(KEY, JSON.stringify(d)); }

function refresh() {
  const d = load();
  let done = 0, total = 0;
  document.querySelectorAll('.card').forEach(c => {
    const k = c.dataset.key;
    const sel = c.querySelector(`input[name="v_${k}"]:checked`);
    const note = c.querySelector('.note');
    if (sel) { done++; }
    if (note.value.trim()) { done++; }
    total += 2;
    c.classList.toggle('done', !!sel);
    if (d[k]) {
      const r = c.querySelector(`input[name="v_${k}"][value="${d[k].v}"]`);
      if (r) r.checked = true;
      note.value = d[k].note || '';
    }
  });
  document.getElementById('progress').textContent =
    `已標 ${done}/${total} 欄`;
}

document.querySelectorAll('.card').forEach(c => {
  const k = c.dataset.key;
  c.querySelectorAll('input[type=radio]').forEach(r =>
    r.addEventListener('change', () => {
      const d = load(); d[k] = { v: r.value, note: c.querySelector('.note').value };
      save(d); refresh();
    }));
  c.querySelector('.note').addEventListener('change', () => {
    const d = load();
    const sel = c.querySelector(`input[name="v_${k}"]:checked`);
    d[k] = { v: sel ? sel.value : '', note: c.querySelector('.note').value };
    save(d); refresh();
  });
});

document.getElementById('export').addEventListener('click', () => {
  const d = load();
  const out = {};
  Object.entries(d).forEach(([k, v]) => {
    const card = document.querySelector(`.card[data-key="${k}"]`);
    if (!card) return;
    const title = card.querySelector('.cap').textContent;
    const cand = card.dataset.cand;
    (out[cand] ||= []).push({ title, fileId: k, verdict: v.v, note: v.note });
  });
  const blob = new Blob([JSON.stringify(out, null, 1)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'orphan-review-decisions.json';
  a.click();
});
document.getElementById('reset').addEventListener('click', () => {
  localStorage.removeItem(KEY); location.reload();
});
refresh();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
