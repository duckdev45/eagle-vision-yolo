"""舊版日報（pptx 進度報告）→ 第三個資料源。

為什麼這批比 QMS 那 36k 張有價值：**它是同一個領域**。
QMS 失敗的原因是稽核照為「查驗點特寫」、日報照為「工人在現場施工」的廣角，
凍結編碼器下完全不遷移（純遷移 top-1 0.283）。這批 pptx 就是同一批工地主任、
同一種相機、同一種構圖，標題也是人手寫的自由文字——與 `data/raw/` 同分佈。
它補的正是現在張數不足被 drop 掉的類別（模板組立、鋼筋綁紮、玻璃…）。

投影片版面高度規則（實測 115.01~115.07 全部一致），所以用**垂直位置**而非
文字內容來認欄位，比 regex 猜穩：

    top≈0.0  頁首「某案進度報告115.01.12」    → 工地名來源，不是標題
    top≈0.6  「12F樑版鋼筋綁紮」                → **工項標題 = 標籤**
    top≈3.9  「115/01/12」×N                    → 日期戳（每張照片一個）
    top≈4.4  「查驗重點: 1.… 2.…」              → 人寫的查驗重點
    top≈5.1  「3」                              → 頁碼

沒有工項標題的投影片一律跳過（進度甘特圖、巡檢表、公工需求、出工統計）——
沒有標題就沒有標籤，留著只是雜訊。

    uv run src/legacy.py --root ~/Downloads/115年日報表 --dry-run   # 只統計不寫檔
    uv run src/legacy.py --root ~/Downloads/115年日報表             # 實際抽出
    uv run src/legacy.py --stats                                    # 對已抽出的算類別分佈
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(__file__))
import paths

# 版面門檻（英吋）。投影片高 5.63"，這些值來自實測而非猜測。
HEADER_MAX_TOP = 0.35  # 這條線以上是頁首
TITLE_MAX_TOP = 1.60  # 標題落在頁首與照片之間；再低就是查驗重點或日期戳
# 標題長度上限。實測最長「9F~3F落地窗玻璃安裝」11 字；放寬到 30 吸收異常寫法，
# 超過就是段落（週工作排程、其他施工概述）而非工項。
TITLE_MAX_LEN = 30
MIN_PIXELS = 200 * 200  # 小於這個的圖是 logo / 圖示，不是工地照

# 不是工項的標題。這些投影片有照片也有短標題，但拍的是表格或文件。
TITLE_JUNK = re.compile(
    r"進度報告|進度匯報|進度表|週工作|工作排程|巡檢表|公工需求|其他施工|其他概述|"
    r"本日工種|本日出工|出工統計|建照|竣工|展延|會議|通知|備忘|附件|目錄"
)
# 純數字/日期/頁碼
NOT_TEXT = re.compile(r"^[\d\s./~年月日:\-]+$")
INSP_HEAD = re.compile(r"^查驗重點\s*[:：]?")
# 檔名或頁首裡的民國日期：115.01.12 / 1150112 / 115.1.12 / 115/01/12
ROC_DATE = re.compile(r"(1[0-2]\d)[./\-]?(\d{1,2})[./\-]?(\d{1,2})")


def dhash(im, size: int = 8) -> str:
    """差分雜湊：縮成 (size+1)×size 灰階，比相鄰像素亮度，得 size² 個 bit。

    為什麼不能只用 sha1：pptx 會把照片重新編碼，同一張照片在 PMS 下載檔與
    投影片裡的位元組完全不同，sha1 對不上。dhash 看的是明暗結構，重壓縮、
    改尺寸都還認得出來。

    ponytail: 只做雜湊完全相等的比對（dict 查表），不算 Hamming 距離。
    差 1~2 bit 的近似張會漏掉；真的要抓那些再上 BK-tree，別為了還沒發生的問題
    先寫 O(n²)。
    """
    import numpy as np

    g = np.asarray(im.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def pms_hashes(log=print) -> dict[str, str]:
    """已下載的 PMS 日報照 → {dhash: fileId}。舊 pptx 撞到就不收。

    同一張照片兩邊都有的話，不擋掉會有兩個後果：訓練集重複計數，
    而且一張進 train、另一張進 test 就是實打實的洩漏。
    """
    from PIL import Image

    if not paths.PHOTOS.exists():
        return {}
    out = {}
    for p in sorted(paths.PHOTOS.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            with Image.open(p) as im:
                out.setdefault(dhash(im), p.stem)
        except Exception:
            continue
    log(f"PMS 既有照片 {len(out)} 種畫面，拿來擋重複")
    return out


def roc_to_iso(y: int, m: int, d: int) -> str | None:
    """民國年 → 西元。115 → 2026。超出合理範圍回 None（手打錯字很常見）。"""
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return None
    return f"{y + 1911:04d}-{m:02d}-{d:02d}"


def date_from_name(text: str) -> str | None:
    """從檔名/資料夾名抓日期。

    **優先用檔名而不是投影片上的日期戳**：實測有投影片打成 `2025/1/12`
    （該案是 2026 年），而檔名 `某案115.1.12` 是對的。檔名是排程產生的，
    日期戳是手打的。
    """
    for m in ROC_DATE.finditer(text):
        iso = roc_to_iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if iso:
            return iso
    return None


# 工地名要對齊 PMS 的 constrName，否則切分時「同工地不跨組」這條鐵律形同虛設
# ——簡稱與全名是同一個案場，不併起來就會一邊進 train 一邊進 test。
# 工地名別名表 = 公司案場資訊，2026-09-01 起外移 reference/site_aliases.yaml（不入 git）。
# 缺檔時不套別名（舊檔名照原樣解析），行為與「沒有別名可套」一致。
try:
    SITE_ALIAS = yaml.safe_load((paths.ROOT / "reference" / "site_aliases.yaml").read_text()) or {}
except FileNotFoundError:
    SITE_ALIAS = {}


def site_from(header: str, fname: str) -> str:
    """工地名。頁首「某案進度報告」→ 某案；抓不到就退檔名去掉日期的部分。"""
    h = re.sub(r"\s+", "", header or "")
    m = re.match(r"([一-鿿\w]{1,8}?)案?(?:進度報告|進度匯報|進度表)", h)
    if m and m.group(1):
        return SITE_ALIAS.get(m.group(1), m.group(1))
    base = re.sub(r"[\d.\-~]+", "", Path(fname).stem)
    base = re.sub(r"進度報告|進度匯報|進度表|進度", "", base)
    base = base.strip("-_ ") or "未知"
    return SITE_ALIAS.get(base, base)


def _iter_shapes(shapes):
    """遞迴展開群組。群組裡的照片與文字都要算，不展開會漏掉整張投影片。"""
    for sh in shapes:
        if sh.shape_type == 6:  # MSO_SHAPE_TYPE.GROUP
            yield from _iter_shapes(sh.shapes)
        else:
            yield sh


def parse_slide(slide, emu_to_in):
    """一張投影片 → (頁首, 工項標題, 查驗重點[], 圖片blob[])。標題為 None 表示不是工項頁。"""
    header, cands, insp, pics = "", [], [], []
    for sh in _iter_shapes(slide.shapes):
        if sh.shape_type == 13:  # PICTURE
            pics.append(sh)
            continue
        if not getattr(sh, "has_text_frame", False):
            continue
        txt = sh.text_frame.text.strip()
        if not txt:
            continue
        top = emu_to_in(sh.top or 0)
        if top <= HEADER_MAX_TOP and not header:
            header = txt
            continue
        if INSP_HEAD.match(txt):
            insp = [
                re.sub(r"^\d+\s*[.、]\s*", "", ln).strip()
                for ln in INSP_HEAD.sub("", txt).splitlines()
                if ln.strip()
            ]
            continue
        if top <= TITLE_MAX_TOP:
            cands.append((top, txt))

    title = None
    for _, txt in sorted(cands):
        one = re.sub(r"\s+", "", txt)
        if (
            len(one) <= TITLE_MAX_LEN
            and "\n" not in txt
            and not NOT_TEXT.match(one)
            and not TITLE_JUNK.search(one)
        ):
            title = one
            break
    return header, title, insp, pics


def run(
    root: str,
    out_dir: Path | None = None,
    limit: int = 0,
    dry: bool = False,
    skip_pms_dupes: bool = True,
    log=print,
) -> dict:
    from PIL import Image
    from pptx import Presentation
    from pptx.util import Emu

    root_p = Path(root).expanduser()
    out = Path(out_dir) if out_dir else paths.LEGACY
    photos = out / "raw" / "photos"
    if not dry:
        photos.mkdir(parents=True, exist_ok=True)
    pms = pms_hashes(log) if skip_pms_dupes else {}

    files = sorted(p for p in root_p.rglob("*.pptx") if not p.name.startswith("~$"))
    if limit:
        files = files[:limit]
    log(f"{len(files)} 個 pptx（{root_p}）")

    rows, seen, byhash, stat = (
        [],
        {},
        {},
        {
            "pptx": len(files),
            "pptxFailed": 0,
            "slides": 0,
            "workItemSlides": 0,
            "photos": 0,
            "dupPhotos": 0,
            "dupVisual": 0,
            "dupWithPms": 0,
            "tinySkipped": 0,
            "noTitleSlides": 0,
        },
    )

    for n, f in enumerate(files, 1):
        try:
            prs = Presentation(str(f))
        except Exception as e:
            log(f"  讀不開 {f.name}: {e}")
            stat["pptxFailed"] += 1
            continue
        # 日期優先序：檔名 → 上層資料夾名（0116~0118 這種區間資料夾抓到第一個日期）
        rdate = date_from_name(f.name) or date_from_name(f.parent.name) or ""
        for si, slide in enumerate(prs.slides, 1):
            stat["slides"] += 1
            header, title, insp, pics = parse_slide(slide, lambda e: Emu(e).inches)
            if not title or not pics:
                stat["noTitleSlides"] += 1 if pics else 0
                continue
            stat["workItemSlides"] += 1
            site = site_from(header, f.name)
            for serial, sh in enumerate(pics, 1):
                try:
                    blob = sh.image.blob
                    ext = (sh.image.ext or "jpg").lower()
                except Exception:
                    continue
                try:
                    im = Image.open(io.BytesIO(blob))
                    w, h = im.size
                except Exception:
                    continue
                if w * h < MIN_PIXELS:
                    stat["tinySkipped"] += 1
                    continue
                fid = hashlib.sha1(blob).hexdigest()
                if fid in seen:
                    # 同一張照片在多份 pptx 出現（單案檔 + 彙整檔）。留第一次，
                    # 但記下來源數量——重複次數高的照片是被反覆引用的代表照。
                    seen[fid] += 1
                    stat["dupPhotos"] += 1
                    continue
                # 位元組不同、畫面相同：pptx 重新編碼過的同一張。sha1 抓不到，dhash 抓得到。
                dh = dhash(im)
                if dh in pms:
                    stat["dupWithPms"] += 1
                    continue
                if dh in byhash:
                    seen[byhash[dh]] += 1
                    stat["dupVisual"] += 1
                    continue
                byhash[dh] = fid
                seen[fid] = 1
                stat["photos"] += 1
                if not dry:
                    dst = photos / f"{fid}.{'jpg' if ext in ('jpeg', 'jpg') else ext}"
                    if not dst.exists():
                        dst.write_bytes(blob)
                rows.append(
                    {
                        "fileId": fid,
                        "source": "legacy",
                        "dhash": dh,
                        "title": title,
                        "inspPoints": json.dumps(insp, ensure_ascii=False) if insp else "",
                        "reportDate": rdate,
                        "constrName": site,
                        "natW": w,
                        "natH": h,
                        "ext": ext,
                        "pptx": str(f.relative_to(root_p)),
                        "slide": si,
                        "serial": serial,
                    }
                )
        if n % 50 == 0:
            log(f"  {n}/{len(files)}  照片 {stat['photos']}")

    for r in rows:
        r["dupCount"] = seen.get(r["fileId"], 1)

    if not dry and rows:
        mf = out / "raw" / "manifest.csv"
        with mf.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        log(f"manifest → {mf}")
        # 去重統計只有抽取時算得出來（manifest 只留下活著的那些）。存起來，
        # 操作台才講得出「擋掉多少、靠什麼擋的」，不然那些數字就只活在 log 裡。
        (out / "raw" / "stats.json").write_text(
            json.dumps({"root": str(root_p), "stat": stat}, ensure_ascii=False, indent=1)
        )
    log(f"統計：{stat}")
    return {"stat": stat, "rows": rows}


def stats(rows=None, log=print) -> dict:
    """套 labels.yaml 看這批補了哪些類別 —— 這才是「值不值得做」的答案。"""
    import collections

    from labels import Labeler

    if rows is None:
        mf = paths.LEGACY_MANIFEST
        if not mf.exists():
            log(f"找不到 {mf}，先跑一次抽取")
            return {}
        rows = list(csv.DictReader(mf.open(encoding="utf-8")))

    lab = Labeler.load()
    cnt = collections.Counter()
    for r in rows:
        cnt[lab.label(r["title"]) or lab.fallback] += 1
    log(f"\n{len(rows)} 張 / {len(cnt)} 類（含 fallback）")
    for k, v in cnt.most_common():
        log(f"  {k:14} {v:5}")

    titles = collections.Counter(r["title"] for r in rows)
    log(f"\n不同標題 {len(titles)} 種，最常見：")
    for k, v in titles.most_common(15):
        log(f"  {v:4}x {k}")
    sites = collections.Counter(r["constrName"] for r in rows)
    log(f"\n工地：{dict(sites)}")
    dates = [r["reportDate"] for r in rows if r["reportDate"]]
    log(
        f"日期範圍：{min(dates) if dates else '-'} ~ {max(dates) if dates else '-'}"
        f"（{len(set(dates))} 天，{len(rows) - len(dates)} 張無日期）"
    )
    return {"byClass": dict(cnt), "titles": len(titles), "sites": dict(sites)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/Downloads/115年日報表")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 個 pptx（試跑用）")
    ap.add_argument("--dry-run", action="store_true", help="只統計，不寫任何檔")
    ap.add_argument("--stats", action="store_true", help="對已抽出的 manifest 算類別分佈")
    ap.add_argument("--keep-pms-dupes", action="store_true", help="不擋與 PMS 畫面重複的照片（預設會擋）")
    a = ap.parse_args()
    if a.stats:
        stats()
    else:
        r = run(a.root, limit=a.limit, dry=a.dry_run, skip_pms_dupes=not a.keep_pms_dupes)
        stats(r["rows"])
