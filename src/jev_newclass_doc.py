"""Jev 政策判定裡「需要人看圖」的那兩類 → 複核活文件。

`jev_newclass_policy.py` 把 19 個候選工種分成六種政策，其中兩種模型明說自己決定不了：

  看圖分桶 (mixed)  —— 這批標題混了多種作業，一概而論會出錯（防火門 32、公設修護 18）
  人工複核 (conf<0.5) —— 信心低於地板，不猜（砌紅磚 30、開挖 22、塔吊 22、消防 20、空調 4）

合計 148 張。其餘 370 張模型有定見（不動 / 維持丟棄 / 寫規則），不佔用你的時間。

與 jev_orphan_doc.py 的差別：
  * 跨 PMS + legacy 兩個來源（上一份只有 legacy），圖片路徑依來源分流
  * 選項改成「判定歸哪一類」而不是四選一固定桶 —— 因為這批的問題是
    「這張到底屬於誰」，不是「要不要留在這桶」
  * 每桶攤開 Jev 的答案與信心，讓你能直接反駁它（鐵律四：人工裁決才是真標籤）

文件不改規則檔。勾選存 localStorage、匯出 JSON，改 labels.yaml 是人的決定。
"""

from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import paths
from core.labeler import Labeler

OUT = paths.REPORTS_OUT / "2026-09-19-jev" / "NEWCLASS-REVIEW.html"
POLICY = paths.REPORTS_OUT / "2026-09-19-jev" / "newclass_policy.csv"

# 每桶的候選歸屬。**全部來自 labels.yaml 既有類別或 QS 樹上的獨立一支**，
# 不自行發明類名（v15 教訓：格柵/扶手看起來像工種，樹上卻散在多處）。
CHOICES: dict[str, list[tuple[str, str]]] = {
    "QS0603 防火門": [
        ("門窗-住戶大門", "門扇本體安裝／站立／分料"),
        ("油漆-批土塗裝", "門框油漆收邊（動作是油漆）"),
        ("防水-矽利康", "門檻矽利康（動作是打膠）"),
        ("門窗-鋁門窗", "框料保護紙撕除等表面處理"),
        ("新類 門窗-防火門", "QS0603 獨立一支，若本體施作夠多"),
    ],
    "QS1002 公共設施修護": [
        ("泥作-地磚貼飾", "人行道／車道鋪面貼磚"),
        ("公共-路緣石", "路緣石、側石"),
        ("雜項-清潔", "道路清掃善後"),
        ("新類 公共-設施修護", "QS1002 獨立一支"),
    ],
    "QS0401 砌紅磚": [
        ("新類 泥作-砌磚", "QS0401 獨立一支，砌磚本體"),
        ("泥作-打底", "標題雙動作時以打底為主"),
        ("輕隔間-灌漿牆", "若實為輕質牆而非紅磚"),
    ],
    "QS0207 開挖": [
        ("基礎-開挖工程", "v15 已寫規則（土方清運／棄土坑）"),
        ("雜項-清潔", "若畫面實為場地清掃"),
        ("基礎-連續壁", "若為連續壁導溝階段"),
    ],
    "QS0105 塔吊": [
        ("新類 假設-塔吊", "QS0105 獨立一支"),
        ("假設-鷹架", "併入假設工程"),
        ("不是施工照", "吊裝作業紀錄／行政"),
    ],
    "QS0903 消防": [
        ("新類 機電-消防", "QS0903 獨立一支"),
        ("防水-防水施作", "若為管道間防水"),
        ("不是施工照", "查驗表單／竣工照"),
    ],
    "QS0904 空調": [
        ("新類 機電-空調", "QS0904 獨立一支"),
        ("門窗-鋁門窗", "冷氣格柵（v15 已裁定歸此）"),
        ("木作-天花封板", "室內機藏天花"),
    ],
}

HINT = {
    "QS0603 防火門": "Jev 判 <b>mixed 信心 0.91</b> —— 標題確實混了四種動作："
    "「防火門安裝」「防火門框割紙」「防火門框油漆收邊」「門檻矽利康」。"
    "跟 v15 扶手 93 張是同一種病：名詞相同、動作不同。<b>以畫面主體為準</b>。",
    "QS1002 公共設施修護": "Jev 判 <b>mixed 信心 0.78</b>。現在 12 張被掃進 泥作-地磚貼飾。"
    "要確認的是：這是建案自己的公設，還是<b>市府要求修復的既有道路</b>？後者 QS1002 有獨立標準。",
    "QS0401 砌紅磚": "Jev <b>信心 0.20，最低的一個</b>，它明說不猜。原因看標題就懂："
    "「1F公設中庭砌磚/打底施作」<b>兩個動作連寫</b>，規則吃到哪個純屬運氣。"
    "QS0401 砌紅磚是獨立一支 —— 但若照片主體都是打底完成面，那就該留在打底。",
    "QS0207 開挖": "Jev <b>信心 0.42 棄權</b>，而你剛拍板寫進規則。這組是<b>驗證題</b>："
    "若你看圖後仍認為是開挖工程，那 v15 的規則就站得住；若多數畫面其實是場地清掃，"
    "那要回頭修 —— 模型的棄權可能在提醒某種真實的模糊。",
    "QS0105 塔吊": "Jev <b>信心 0.34 棄權</b>。16 張現在落在 fallback「其他」。"
    "QS0105 施工塔吊有獨立標準，但要確認畫面是<b>塔吊本體組裝／爬升</b>還是只是入鏡的背景。",
    "QS0903 消防": "Jev <b>信心 0.41 棄權</b>。12 張被規則丟棄、6 張歸防水。"
    "消防是機電分包，labels.yaml 目前<b>完全沒有機電類別</b> —— 開這一支等於開新領域，"
    "要想清楚視覺軸分不分得出來。",
    "QS0904 空調": "Jev <b>信心 0.36 棄權</b>，只有 4 張。"
    "注意 v15 已裁定<b>冷氣格柵歸門窗-鋁門窗</b>（QS0602 含冷氣隔柵），別重複開類。",
}


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def img_path(row) -> Path | None:
    p = (paths.IMAGES if row.src == "pms" else paths.LEGACY_IMAGES) / f"{row.fileId}.jpg"
    return p if p.exists() else None


def card(row, bucket: str, idx: int, total: int) -> str:
    p = img_path(row)
    if p is None:
        return ""
    title = html.escape(str(row.title))
    site = html.escape(str(row.get("constrName") or ""))
    opts = "".join(
        f'<label title="{html.escape(why)}">'
        f'<input type="radio" name="v_{row.fileId}" value="{html.escape(lab)}"> {html.escape(lab)}</label>'
        for lab, why in CHOICES[bucket]
    )
    return f"""
<div class="card" data-key="{row.fileId}" data-bucket="{html.escape(bucket)}">
  <div class="imgwrap"><img loading="lazy" src="data:image/jpeg;base64,{b64(p)}" alt="{title}"></div>
  <div class="cap">{title}</div>
  <div class="meta">{html.escape(str(row.現歸))} · {site} · {row.src} · {idx + 1}/{total}</div>
  <div class="opts">{opts}</div>
  <input class="note" placeholder="看圖備註…">
</div>"""


def main() -> int:
    pol = pd.read_csv(POLICY)
    need = pol[pol.policy.isin(["看圖分桶", "人工複核"])].copy()

    lab = Labeler.load()
    pms = lab.drop_excluded(pd.read_csv(paths.MANIFEST))
    pms = pms.assign(cls=pms.title.map(lab.label), src="pms")
    lg = pd.read_csv(paths.LEGACY_MANIFEST, dtype=str)
    lg = lg.assign(cls=lg.title.astype(str).map(lab.label), src="legacy")
    if "constrName" not in lg:
        lg["constrName"] = ""
    cols = ["fileId", "title", "cls", "src", "constrName"]
    al = pd.concat([pms[cols], lg[cols]], ignore_index=True)

    probe = yaml.safe_load((paths.ROOT / "data" / "qs_probe.yaml").read_text())

    sections, n_photo = [], 0
    for _, prow in need.iterrows():
        b = prow["工種"]
        if b not in CHOICES:
            print(f"  ⚠ {b} 沒有候選清單，跳過（要先想清楚它能歸哪，不要臨時發明類名）")
            continue
        hit = al[al.title.astype(str).str.contains(probe[b]["pattern"], na=False, regex=True)].copy()
        hit["現歸"] = hit.cls.fillna("（規則丟棄）")
        hit = hit.sort_values(["src", "title"])
        cards = "".join(card(r, b, i, len(hit)) for i, (_, r) in enumerate(hit.iterrows()))
        n_photo += len(hit)
        verdict = (
            f"Jev：{prow['same_work']} · 信心 {prow['confidence']:.2f} · "
            f"非施工 {prow['not_construction']:.2f} · <b>{prow['policy']}</b>"
        )
        sections.append(
            f"""
<section>
  <h2>{html.escape(b)} <span class="cnt">{len(hit)} 張 / {hit.title.nunique()} 種標題</span></h2>
  <div class="jev">{verdict}</div>
  <div class="hint">{HINT.get(b, "")}</div>
  <div class="grid">{cards}</div>
</section>"""
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        DOC.replace("{{SECTIONS}}", "".join(sections))
        .replace("{{N}}", str(n_photo))
        .replace("{{NB}}", str(len(sections))),
        encoding="utf-8",
    )
    print(f"{n_photo} 張 / {len(sections)} 桶 → {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


DOC = """<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>新類別複核 — Eagle Vision</title>
<style>
:root { color-scheme: light dark; }
body { font-family: var(--font, -apple-system, "PingFang TC", sans-serif);
       color: var(--foreground, #222); background: var(--card, #fff);
       margin: 0; padding: 16px; }
header { position: sticky; top: 0; background: var(--card, #fff);
         padding: 8px 0; border-bottom: 1px solid var(--border, #ddd); z-index: 9; }
h1 { font-size: 18px; margin: 4px 0; }
h2 { font-size: 16px; margin: 28px 0 4px; }
.cnt { font-weight: 400; opacity: .65; font-size: 13px; }
.jev { font-size: 12px; opacity: .75; margin: 2px 0 6px;
       font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.hint { font-size: 13px; opacity: .85; margin: 4px 0 12px; max-width: 74ch;
        line-height: 1.65; border-left: 3px solid var(--accent, #08c);
        padding-left: 10px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 12px; }
.card { border: 1px solid var(--border, #ccc); border-radius: 8px;
        padding: 8px; background: var(--card, #fff); }
.imgwrap { aspect-ratio: 4/3; overflow: hidden; border-radius: 4px; }
.imgwrap img { width: 100%; height: 100%; object-fit: cover; cursor: zoom-in;
               transition: transform .2s; }
.imgwrap img:hover { transform: scale(1.04); }
.cap { font-size: 13px; margin: 6px 0 2px; line-height: 1.4; }
.meta { font-size: 11px; opacity: .55; }
.opts { display: flex; flex-direction: column; gap: 3px;
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
footer { margin: 32px 0 8px; font-size: 12px; opacity: .55; line-height: 1.7; }
</style></head><body>
<header>
  <h1>新類別複核 — Jev 說它決定不了的那些</h1>
  <div style="font-size:12px;opacity:.7">
    {{N}} 張 · {{NB}} 桶 · 模型有定見的 370 張不在這裡 · 勾選存本機 · 匯出 JSON 後再改 labels.yaml
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
  候選歸屬全部來自 labels.yaml 既有類別或 QS 樹上的獨立一支，沒有臨時發明的類名。<br>
  「新類 ○○」選項代表 QS 樹上確實獨立一支 —— 但仍受 min_class_size=12 管制，
  張數不足會寫規則等放行（同 v13 連續壁先例）。<br>
  產生器：src/jev_newclass_doc.py
</footer>
<script>
const KEY = 'newclass-review-v1';
document.querySelectorAll('.imgwrap img').forEach(img => {
  img.addEventListener('click', () => {
    document.getElementById('lbimg').src = img.src;
    document.getElementById('lightbox').classList.add('show');
  });
});
document.getElementById('lightbox').addEventListener('click', e =>
  e.currentTarget.classList.remove('show'));

const load = () => { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); }
                     catch { return {}; } };
const save = d => localStorage.setItem(KEY, JSON.stringify(d));

function refresh() {
  const d = load();
  let done = 0;
  const cards = document.querySelectorAll('.card');
  cards.forEach(c => {
    const k = c.dataset.key;
    if (d[k]) {
      const r = c.querySelector(`input[name="v_${k}"][value="${CSS.escape(d[k].v)}"]`);
      if (r) r.checked = true;
      c.querySelector('.note').value = d[k].note || '';
    }
    const sel = c.querySelector(`input[name="v_${k}"]:checked`);
    if (sel) done++;
    c.classList.toggle('done', !!sel);
  });
  document.getElementById('progress').textContent = `已判 ${done}/${cards.length} 張`;
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
  const d = load(), out = {};
  Object.entries(d).forEach(([k, v]) => {
    const card = document.querySelector(`.card[data-key="${k}"]`);
    if (!card) return;
    (out[card.dataset.bucket] ||= []).push({
      title: card.querySelector('.cap').textContent,
      fileId: k, verdict: v.v, note: v.note });
  });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)],
                                        { type: 'application/json' }));
  a.download = 'newclass-review-decisions.json';
  a.click();
});
document.getElementById('reset').addEventListener('click', () => {
  localStorage.removeItem(KEY); location.reload();
});
refresh();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
