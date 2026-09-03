"""三個公開缺失資料集的「自己的照片＋自己的標註」預覽板（tab 切換 + 分頁 + 放大版）。

- 三個資料集各一個 tab 按鈕，切換顯示
- 每集抽 N 張（--n，預設 300），每頁顯示 --page-size 張（預設 60），可翻頁
- 縮圖／放大圖都是外部檔案（thumb/、full/），index.html 只放 metadata——
  N 再大也不會撐爆單一 HTML（上一版全塞 base64，60 張/集就 30MB，
  不限制会到 ~1GB 直接讓瀏覽器卡死）
- 點圖放大（lightbox），←→ 翻頁、Esc/點背景關閉

    uv run src/fr_dataset_preview.py [--n 300] [--page-size 60]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path("/Users/duck/PycharmProjects/eagle-vision/reports/yolo-smoke/dataset-preview")

MBDD_NAMES = {0: "crack", 1: "leakage", 2: "abscission", 3: "corrosion", 4: "bulge"}
MBDD_COLORS = {0: (225, 29, 72), 1: (14, 165, 233), 2: (249, 115, 22), 3: (124, 58, 237), 4: (5, 150, 105)}

GYU_NAMES = {0: "Seepage", 1: "cracks", 2: "exposed_rebar", 3: "honeycomb_surface", 4: "rust", 5: "spalling"}
GYU_COLORS = {0: (14, 165, 233), 1: (225, 29, 72), 2: (250, 204, 21), 3: (168, 85, 247), 4: (249, 115, 22),
              5: (5, 150, 105)}


def save_pair(path: Path, draw_fn, thumb_p: Path, full_p: Path,
              thumb_side: int = 320, full_side: int = 1600) -> None:
    """存縮圖＋放大圖兩個外部檔案（都是「先縮再畫」，框線寬度才不會被縮圖吃掉）。"""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((thumb_side, thumb_side))
        W, H = im.size
        draw_fn(ImageDraw.Draw(im, "RGBA"), W, H)
        im.save(thumb_p, "JPEG", quality=80)
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((full_side, full_side))
        W, H = im.size
        draw_fn(ImageDraw.Draw(im, "RGBA"), W, H, big=True)
        im.save(full_p, "JPEG", quality=88)


def poly_to_points(pts, W, H):
    return [(float(pts[i]) * W, float(pts[i + 1]) * H) for i in range(0, len(pts), 2)]


def draw_tag(dr, x1, y1, text, color, fs):
    """實心底色標籤貼在框頂端，白字——參考 Grounding DINO 那種 class+conf 貼牌風格。"""
    pad = max(2, fs // 5)
    box = dr.textbbox((0, 0), text, font_size=fs)
    tw, th = box[2] - box[0], box[3] - box[1]
    tag_h = th + 2 * pad
    ty = y1 - tag_h if y1 - tag_h >= 0 else y1
    dr.rectangle([x1, ty, x1 + tw + 2 * pad, ty + tag_h], fill=color)
    dr.text((x1 + pad, ty + pad - box[1]), text, fill=(255, 255, 255), font_size=fs)


def pick_crack_seg(n):
    root = Path("/Users/duck/datasets/crack-seg")
    files = sorted((root / "images" / "train").glob("*"))
    rng = random.Random(7)
    pool = [f for f in files if (root / "labels" / "train" / (f.stem + ".txt")).exists()]
    n = min(n, len(pool))
    return rng.sample(pool, n), root


def draw_crack_seg(f: Path, root: Path):
    lbl = root / "labels" / "train" / (f.stem + ".txt")

    def draw(dr, W, H, big=False):
        w = 6 if big else 3
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 8:
                pts = poly_to_points(p[1:], W, H)
                dr.polygon(pts, fill=(225, 29, 72, 50), outline=(225, 29, 72, 255), width=w)

    return draw, f"crack-seg · {f.stem[:22]}"


def pick_mbdd(n):
    root = Path("/Users/duck/datasets/MBDD2025/MBDD2025")
    rng = random.Random(11)
    ids = [l.strip() for l in (root / "train.txt").read_text().splitlines() if l.strip()]
    labeled = []
    for rel in ids:
        lbl_p = root / "Labels" / (Path(rel).stem + ".txt")
        if not lbl_p.exists():
            continue
        for line in lbl_p.read_text().splitlines():
            if len(line.split()) == 5:
                labeled.append(rel)
                break
    n = min(n, len(labeled))
    return rng.sample(labeled, n), root


def draw_mbdd(rel: str, root: Path):
    img_p = root / "JPEGImages" / Path(rel).name
    lbl_p = root / "Labels" / (Path(rel).stem + ".txt")

    def draw(dr, W, H, big=False):
        w = 6 if big else 2
        fs = 22 if big else 11
        for line in lbl_p.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                cls, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
                x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
                c = MBDD_COLORS.get(cls, (100, 100, 100))
                dr.rectangle([x1, y1, x2, y2], outline=c, width=w)
                draw_tag(dr, x1, y1, MBDD_NAMES.get(cls, str(cls)), c, fs)

    return img_p, draw, f"MBDD2025 · {Path(rel).stem}"


def pick_gyu(n):
    root = Path("/Users/duck/datasets/gyu-det/train")
    files = sorted((root / "images").glob("*"))
    rng = random.Random(19)
    pool = [f for f in files if (root / "labels" / (f.stem + ".txt")).exists()]
    n = min(n, len(pool))
    return rng.sample(pool, n), root


def draw_gyu(f: Path, root: Path):
    lbl = root / "labels" / (f.stem + ".txt")

    def draw(dr, W, H, big=False):
        w = 6 if big else 2
        fs = 22 if big else 11
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                cls, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
                x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
                c = GYU_COLORS.get(cls, (100, 100, 100))
                dr.rectangle([x1, y1, x2, y2], outline=c, width=w)
                draw_tag(dr, x1, y1, GYU_NAMES.get(cls, str(cls)), c, fs)

    return draw, f"gyu-det · {f.stem[:22]}"


def pick_tile(n):
    root = Path("/Users/duck/datasets/tile-damage/damage_detection.v85i.yolov7/train/images")
    files = sorted(root.glob("*.jpg"))
    rng = random.Random(3)
    n = min(n, len(files))
    return rng.sample(files, n), root


HTML = """<!doctype html><meta charset="utf-8"><title>dataset preview</title>
<style>
body{font-family:system-ui;margin:24px;background:#0f1115;color:#e5e7eb}
h2{margin:0 0 4px;font-size:18px}
p.cap{margin:0 0 12px;color:#9ca3af;font-size:13px}
.tabs{display:flex;gap:8px;margin-bottom:18px;border-bottom:1px solid #262b36}
.tab{background:none;border:none;color:#9ca3af;font-size:14px;padding:10px 16px;cursor:pointer;
     border-bottom:2px solid transparent;font-family:system-ui}
.tab.active{color:#e5e7eb;border-bottom-color:#e11d48}
.tab .n{color:#6b7280;font-size:12px;margin-left:4px}
.panel{display:none}
.panel.active{display:block}
.pager{display:flex;align-items:center;gap:10px;margin:4px 0 14px;font-size:13px;color:#9ca3af}
.pager button{background:#1c2029;border:1px solid #2c3140;color:#e5e7eb;border-radius:6px;
              padding:5px 12px;cursor:pointer;font-size:13px}
.pager button:disabled{opacity:.35;cursor:default}
.pager .pg{font-family:ui-monospace}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;min-height:200px}
.card{background:#161a22;border-radius:10px;padding:7px;cursor:zoom-in;transition:transform .1s}
.card:hover{transform:scale(1.02)}
.card img{width:100%;border-radius:6px;display:block;aspect-ratio:1;object-fit:cover;background:#1c2029}
.card div{font-size:10.5px;color:#9ca3af;padding-top:5px;font-family:ui-monospace;
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#lb{position:fixed;inset:0;background:#000c;z-index:9999;display:none;align-items:center;
    justify-content:center;flex-direction:column;cursor:zoom-out}
#lb img{max-width:96vw;max-height:88vh;border-radius:8px;cursor:default}
#lb .cap{color:#e5e7eb;margin-top:10px;font-family:ui-monospace;font-size:13px}
#lb .nav{color:#e5e7eb;user-select:none;position:absolute;top:50%;font-size:44px;
         padding:8px 18px;cursor:pointer;transform:translateY(-50%);background:#0006;border-radius:8px}
#lb .prev{left:16px} #lb .next{right:16px}
</style>
__TABS__
__PANELS__
<div id="lb"><span class="nav prev" onclick="event.stopPropagation();lbStep(-1)">&#10094;</span>
<img id="lbi"><div class="cap" id="lbc"></div>
<span class="nav next" onclick="event.stopPropagation();lbStep(1)">&#10095;</span></div>
<script>
const DATA = __DATA__;      // { key: [{thumb,full,label}, ...], ... }
const PAGE_SIZE = __PAGE_SIZE__;
const page = {};            // key -> 目前頁碼（0-based）
let curKey = null, curFlat = [], curIdxInPage = 0;

function showTab(key){
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active', p.dataset.name===key));
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active', t.dataset.name===key));
  curKey = key;
  if(!(key in page)) page[key] = 0;
  renderPage(key);
}

function totalPages(key){ return Math.max(1, Math.ceil(DATA[key].length / PAGE_SIZE)); }

function renderPage(key){
  const p = page[key];
  const start = p * PAGE_SIZE, end = Math.min(start + PAGE_SIZE, DATA[key].length);
  const slice = DATA[key].slice(start, end);
  const grid = document.getElementById('grid-' + key);
  grid.innerHTML = slice.map((c, i) => `
    <div class="card" onclick="lbOpenFrom('${key}', ${start + i})">
      <img loading="lazy" src="${c.thumb}">
      <div>${c.label}</div>
    </div>`).join('');
  const tp = totalPages(key);
  document.getElementById('pg-' + key).textContent = (p + 1) + ' / ' + tp +
    '　(' + (start + 1) + '–' + end + ' / ' + DATA[key].length + ' 張)';
  document.getElementById('prev-' + key).disabled = p <= 0;
  document.getElementById('next-' + key).disabled = p >= tp - 1;
}

function gotoPage(key, delta){
  const tp = totalPages(key);
  page[key] = Math.min(Math.max(0, page[key] + delta), tp - 1);
  renderPage(key);
}

function lbOpenFrom(key, globalIdx){
  curFlat = DATA[key]; curIdxInPage = globalIdx;
  document.getElementById('lb').style.display = 'flex';
  paintLb();
}
function paintLb(){
  const c = curFlat[curIdxInPage];
  document.getElementById('lbi').src = c.full;
  document.getElementById('lbc').textContent = c.label + '  (' + (curIdxInPage + 1) + '/' + curFlat.length + ')';
}
function lbStep(d){
  curIdxInPage = (curIdxInPage + d + curFlat.length) % curFlat.length;
  // 跨頁：如果翻出了目前頁的範圍，順便切頁面顯示，避免縮圖格跟大圖對不上
  const p = Math.floor(curIdxInPage / PAGE_SIZE);
  if(p !== page[curKey]){ page[curKey] = p; renderPage(curKey); }
  paintLb();
}
function lbClose(){ document.getElementById('lb').style.display = 'none'; }
document.addEventListener('keydown', e=>{
  if(document.getElementById('lb').style.display !== 'flex') return;
  if(e.key === 'ArrowRight') lbStep(1);
  if(e.key === 'ArrowLeft') lbStep(-1);
  if(e.key === 'Escape') lbClose();
});
document.getElementById('lb').addEventListener('click', e=>{ if(e.target.id === 'lb') lbClose(); });

showTab(Object.keys(DATA)[0]);
</script>
</body></html>"""


def build(secs_meta, page_size):
    tabs = ['<div class="tabs">']
    panels = []
    data = {}
    for key, title, cap, cards in secs_meta:
        tabs.append(f'<button class="tab" data-name="{key}" onclick="showTab(\'{key}\')">'
                     f'{key}<span class="n">({len(cards)})</span></button>')
        panels.append(f'''<div class="panel" data-name="{key}">
<h2>{title}</h2><p class="cap">{cap}</p>
<div class="pager">
  <button id="prev-{key}" onclick="gotoPage('{key}',-1)">&#8592; 上一頁</button>
  <span class="pg" id="pg-{key}"></span>
  <button id="next-{key}" onclick="gotoPage('{key}',1)">下一頁 &#8594;</button>
</div>
<div class="grid" id="grid-{key}"></div>
</div>''')
        data[key] = cards
    tabs.append("</div>")
    html = (HTML.replace("__TABS__", "".join(tabs))
                .replace("__PANELS__", "".join(panels))
                .replace("__DATA__", json.dumps(data))
                .replace("__PAGE_SIZE__", str(page_size)))
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="每個資料集抽樣張數（預設 300）")
    ap.add_argument("--page-size", type=int, default=60, help="每頁顯示張數（預設 60）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    thumb_dir = OUT / "thumb"
    full_dir = OUT / "full"
    thumb_dir.mkdir(exist_ok=True)
    full_dir.mkdir(exist_ok=True)

    secs = [
        ("crack-seg", "crack-seg（Ultralytics · 4,029 張）— 分割多邊形 GT",
         "近拍裂縫；紅色半透明=像素級裂縫範圍", pick_crack_seg),
        ("MBDD2025", "MBDD2025（UAV · 14,471 張）— bbox GT · 5 類",
         "空拍牆面：crack/leakage/abscission/corrosion/bulge", pick_mbdd),
        ("gyu-det", "gyu-det（橋梁 · 7,711 張）— bbox GT · 6 類",
         "近拍橋梁：Seepage/cracks/exposed_rebar/honeycomb_surface/rust/spalling"
         "（蜂窩+裸筋唯一來源）", pick_gyu),
        ("tile-damage", "tile-damage（Mendeley · 5,080 張）— 無標註",
         "磁磚裂/坑/剝失/釉面剝落；純素材池，標註要自己做", pick_tile),
    ]

    est_total = args.n * len(secs)
    if est_total > 1000:
        print(f"⚠ 共 {est_total} 張，預估需要數分鐘、磁碟約 {est_total * 140 / 1024:.0f}MB。")

    secs_meta = []
    t0 = time.time()
    for key, title, cap, picker in secs:
        items, root = picker(args.n)
        cards = []
        for idx, item in enumerate(items):
            gid = f"{key}-{idx}"
            thumb_p = thumb_dir / f"{gid}.jpg"
            full_p = full_dir / f"{gid}.jpg"
            if key == "crack-seg":
                draw, label = draw_crack_seg(item, root)
                save_pair(item, draw, thumb_p, full_p)
            elif key == "MBDD2025":
                img_p, draw, label = draw_mbdd(item, root)
                save_pair(img_p, draw, thumb_p, full_p)
            elif key == "gyu-det":
                draw, label = draw_gyu(item, root)
                save_pair(item, draw, thumb_p, full_p)
            else:
                save_pair(item, lambda dr, W, H, big=False: None, thumb_p, full_p)
                label = f"tile-damage · {item.stem[:22]}（無 GT）"
            cards.append({"thumb": f"thumb/{gid}.jpg", "full": f"full/{gid}.jpg", "label": label})
        secs_meta.append((key, title, cap, cards))
        print(f"  {key}: {len(cards)} 張完成（累計 {time.time()-t0:.0f}s）")

    html = build(secs_meta, args.page_size)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    total = sum(len(c) for _, _, _, c in secs_meta)
    print(f"\nwrote {OUT / 'index.html'} | 共 {total} 張 | 每頁 {args.page_size} 張 | 耗時 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
