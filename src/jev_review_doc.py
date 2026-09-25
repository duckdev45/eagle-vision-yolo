"""Jev 探測結果 → 看圖複核活文件（單一 HTML，照片內嵌）。

v11 的紀律是「看圖確認才收編」。這支把 Jev 的提案排版成可以**一邊看圖一邊勾**的東西，
產出兩份清單：

  ① 規則補丁提案　Jev 高信心、regex 沒意見的標題 → 該補進哪條 labels.yaml 規則
  ② 真標籤矛盾　　同一標題被人工標成多個類別 → 要人看圖裁定誰對

勾選狀態存 localStorage，決定匯出成 JSON（收編/退件/待查），不直接改 labels.yaml——
改規則是人的決定，這份文件只負責把證據擺到眼前。

    uv run src/jev_review_doc.py                      # 讀最新的 reports/*-jev/
    uv run src/jev_review_doc.py --run 2026-09-19-jev
    uv run src/jev_review_doc.py --per-title 4        # 每個標題最多放幾張圖

照片是 base64 內嵌（縮到長邊 320），所以單一檔案可以直接丟給別人看，不必附資料夾。
公司照片不進 git：產物寫到 reports/ 底下，該目錄已在 .gitignore。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
from core.labeler import Labeler

THUMB = 320  # 長邊；夠看清楚工種畫面，又不會讓單檔爆掉


def thumb_b64(path: Path) -> str | None:
    from PIL import Image

    try:
        im = Image.open(path)
        im.thumbnail((THUMB, THUMB))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=72)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def photo_index() -> pd.DataFrame:
    """title → fileId → 本機縮圖路徑。日報與 legacy 兩套命名，分開解析後併起來。"""
    out = []
    daily = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
    daily = daily.assign(img=daily.fileId.map(lambda f: paths.IMAGES / f"{f}.jpg"), src="daily")
    out.append(daily[["fileId", "title", "img", "src"]])
    if paths.LEGACY_MANIFEST.exists():
        lg = pd.read_csv(paths.LEGACY_MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
        lg = lg.assign(img=lg.fileId.map(lambda f: paths.LEGACY_IMAGES / f"{f}.jpg"), src="legacy")
        out.append(lg[["fileId", "title", "img", "src"]])
    return pd.concat(out, ignore_index=True)


def rule_for(cls: str) -> str:
    """該類別現行的 regex，讓人一眼看出「要加的關鍵字擺哪」。"""
    import yaml

    cfg = yaml.safe_load(paths.LABELS_YAML.read_text())
    for r in cfg.get("rules") or []:
        if r["label"] == cls:
            return r["pattern"]
    return "（此類無規則）"


def build(run: str, per_title: int) -> Path:
    outdir = paths.ROOT / "reports" / run
    probe = pd.read_csv(outdir / "fallback_probe.csv")
    metrics = json.loads((outdir / "metrics.json").read_text())
    lab = Labeler.load()
    idx = photo_index()

    # ① 規則補丁提案：只看「regex 真的沒意見」的（被 junk 主動丟掉的不算——
    #    那些 regex 有意見，而且 Jev 也同意該丟）。
    probe = probe.assign(rx=probe.title.map(lab.label))
    cand = probe[(probe.rx == lab.fallback) & (probe.jev != "其他") & (probe.conf >= 0.9)].sort_values(
        "photos", ascending=False
    )

    items = []
    for _, r in cand.iterrows():
        shots = idx[idx.title == r.title].head(per_title)
        imgs = [b for b in (thumb_b64(Path(p)) for p in shots.img) if b]
        items.append(
            {
                "title": r.title,
                "photos": int(r.photos),
                "src": r.src,
                "jev": r.jev,
                "conf": float(r.conf),
                "rule": rule_for(r.jev),
                "imgs": imgs,
            }
        )

    # ② 真標籤矛盾：同一標題被標成多類，逐張列出各自的裁決
    rv = pd.read_csv(paths.REVIEW, dtype=str).drop_duplicates("fileId", keep="last")
    conflicts = []
    for c in metrics.get("groundTruthConflicts", []):
        shots = idx[idx.title == c["title"]].merge(rv[["fileId", "cls"]], on="fileId", how="left")
        conflicts.append(
            {
                "title": c["title"],
                "labels": c["labels"],
                "shots": [
                    {"cls": s.cls, "img": thumb_b64(Path(s.img))}
                    for s in shots.itertuples()
                    if pd.notna(s.cls)
                ],
            }
        )

    html = render(items, conflicts, metrics, run)
    out = outdir / "REVIEW.html"
    out.write_text(html, encoding="utf-8")
    return out


def render(items: list, conflicts: list, metrics: dict, run: str) -> str:
    by_class: dict[str, list] = {}
    for it in items:
        by_class.setdefault(it["jev"], []).append(it)
    groups = sorted(by_class.items(), key=lambda kv: -sum(i["photos"] for i in kv[1]))

    bt = metrics.get("byTitle", {})
    summary = "".join(
        f"<tr><td>{k}</td><td class=n>{v['top1']}</td>"
        f"<td class=n>{v['coverage@p90']}</td><td class=n>{v['abstain']}/{v['support']}</td></tr>"
        for k, v in bt.items()
    )
    paired = metrics.get("paired", {}).get("regex_vs_jev", {})

    sections = []
    for cls, rows in groups:
        n_photos = sum(r["photos"] for r in rows)
        cards = []
        for r in rows:
            imgs = "".join(f'<img src="data:image/jpeg;base64,{b}">' for b in r["imgs"])
            cards.append(f"""
        <div class=card data-title="{r["title"]}">
          <div class=head>
            <label><input type=checkbox class=ck data-t="{r["title"]}" data-c="{cls}"><b>{r["title"]}</b></label>
            <span class=meta>{r["photos"]} 張 · {r["src"]} · conf {r["conf"]:.2f}</span>
          </div>
          <div class=shots>{imgs or "<i>（無縮圖）</i>"}</div>
        </div>""")
        sections.append(f"""
      <section>
        <h3>{cls} <span class=meta>{len(rows)} 種寫法 · {n_photos} 張</span></h3>
        <pre class=rule>{rows[0]["rule"]}</pre>
        {"".join(cards)}
      </section>""")

    conf_html = []
    for c in conflicts:
        shots = "".join(
            f'<figure><img src="data:image/jpeg;base64,{s["img"]}"><figcaption>{s["cls"]}</figcaption></figure>'
            for s in c["shots"]
            if s["img"]
        )
        conf_html.append(f"""
      <section class=conflict>
        <h3>{c["title"]}</h3>
        <p class=meta>人工裁決分歧：{" / ".join(c["labels"])}</p>
        <div class=gallery>{shots}</div>
      </section>""")

    return f"""<!doctype html><html lang=zh-Hant><meta charset=utf-8>
<title>Jev 複核 · {run}</title>
<style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;padding:24px;font:14px/1.6 -apple-system,"Noto Sans TC",sans-serif;
background:#14161a;color:#e6e6e6;max-width:1100px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 10px;
border-bottom:1px solid #2a2e35;padding-bottom:6px}} h3{{font-size:14px;margin:16px 0 6px}}
.meta{{color:#8b93a0;font-weight:400;font-size:12px}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{padding:4px 12px;border-bottom:1px solid #2a2e35;text-align:left}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.rule{{background:#1b1e24;padding:6px 10px;border-radius:4px;font-size:12px;color:#9ad;
overflow-x:auto;white-space:pre-wrap;word-break:break-all;margin:4px 0 10px}}
.card{{border:1px solid #2a2e35;border-radius:6px;padding:10px;margin:8px 0;background:#181b20}}
.card.done{{opacity:.45}}
.head{{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}}
.shots{{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}}
.shots img{{height:130px;border-radius:4px;cursor:zoom-in}}
.shots img.zoom{{height:auto;max-width:100%}}
.gallery{{display:flex;gap:10px;flex-wrap:wrap}}
figure{{margin:0;text-align:center}} figure img{{height:150px;border-radius:4px}}
figcaption{{font-size:12px;color:#8b93a0;margin-top:2px}}
.conflict{{border-left:3px solid #c96;padding-left:12px}}
.bar{{position:sticky;top:0;background:#14161a;padding:10px 0;border-bottom:1px solid #2a2e35;
margin-bottom:12px;z-index:9;display:flex;gap:10px;align-items:center}}
button{{background:#2a2e35;color:#e6e6e6;border:0;padding:6px 14px;border-radius:4px;cursor:pointer;font:inherit}}
button:hover{{background:#363b44}}
.note{{background:#1b1e24;border-left:3px solid #4a7;padding:10px 14px;margin:12px 0;font-size:13px}}
</style>
<h1>Jev 複核清單 · {run}</h1>
<p class=meta>labels.yaml v{metrics.get("labelsVersion")} · {metrics.get("model")} ·
split {metrics.get("split")}</p>

<h2>A/B 結果（標題層級）</h2>
<table><tr><th>arm<th>top-1<th>coverage@p90<th>棄權</tr>{summary}</table>
<p class=meta>McNemar：regex 贏 {paired.get("aOnly")} 筆，Jev 贏 {paired.get("bOnly")} 筆，
p={paired.get("p")}。樣本 {metrics.get("titles")} 種標題／{metrics.get("photos")} 張照片。</p>

<div class=note>
<b>這份文件不會改 labels.yaml。</b>勾選＝你看圖確認過該收編，按「匯出決定」拿 JSON，
規則要怎麼寫仍是人的決定。依 v11 紀律：沒看過圖的不收編。
</div>

<div class=bar>
  <button onclick="exportJSON()">匯出決定 JSON</button>
  <button onclick="document.querySelectorAll('.ck').forEach(c=>{{c.checked=true;c.dispatchEvent(new Event('change'))}})">全選</button>
  <button onclick="document.querySelectorAll('.ck').forEach(c=>{{c.checked=false;c.dispatchEvent(new Event('change'))}})">全清</button>
  <span class=meta id=tally></span>
</div>

<h2>① 規則補丁提案（{len(items)} 種寫法）</h2>
<p class=meta>regex 沒有意見、Jev 信心 ≥ 0.90。點圖可放大。</p>
{"".join(sections)}

<h2>② 真標籤矛盾（{len(conflicts)} 組）</h2>
<p class=meta>同一標題被人工標成不同類別。照片層級裁決本來就可能與標題不同
（一個工項的多張照片常在拍不同階段），要看圖才知道是分歧還是手滑。</p>
{"".join(conf_html)}

<script>
const KEY='jev-review-{run}';
const saved=JSON.parse(localStorage.getItem(KEY)||'{{}}');
document.querySelectorAll('.ck').forEach(c=>{{
  if(saved[c.dataset.t]) c.checked=true;
  c.closest('.card').classList.toggle('done',c.checked);
  c.addEventListener('change',()=>{{
    c.closest('.card').classList.toggle('done',c.checked);
    const s=JSON.parse(localStorage.getItem(KEY)||'{{}}');
    c.checked ? s[c.dataset.t]=c.dataset.c : delete s[c.dataset.t];
    localStorage.setItem(KEY,JSON.stringify(s)); tally();
  }});
}});
document.querySelectorAll('.shots img').forEach(i=>
  i.addEventListener('click',()=>i.classList.toggle('zoom')));
function tally(){{
  const n=document.querySelectorAll('.ck:checked').length;
  document.getElementById('tally').textContent=`已勾 ${{n}} / {len(items)}`;
}}
function exportJSON(){{
  const out={{}};
  document.querySelectorAll('.ck').forEach(c=>{{
    (out[c.dataset.c]=out[c.dataset.c]||[]).push({{title:c.dataset.t,收編:c.checked}});
  }});
  const b=new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b); a.download='jev-review-decisions.json'; a.click();
}}
tally();
</script>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="reports/ 底下的目錄名，預設取最新的 *-jev")
    ap.add_argument("--per-title", type=int, default=4)
    a = ap.parse_args()

    run = a.run
    if not run:
        runs = sorted(p.name for p in (paths.ROOT / "reports").glob("*-jev"))
        if not runs:
            print("找不到 reports/*-jev，先跑 make jev-eval 與 make jev-fallback", file=sys.stderr)
            return 2
        run = runs[-1]

    out = build(run, a.per_title)
    mb = out.stat().st_size / 1e6
    print(f"複核文件 → {out}  ({mb:.1f} MB)")
    print(f"開啟：open {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
