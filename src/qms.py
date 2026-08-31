"""QMS 稽核照同步（另一個系統、另一套標籤）。

與日報那套**完全分開**：不同 base URL、不同帳號、不同資料夾、不同 manifest。
兩邊唯一共用的是 prepare.py 的遮蔽規則——要合併訓練就必須遮得一模一樣，
否則「左下有沒有黑塊」本身就會變成區分兩個資料源的洩漏訊號。

標籤來源比日報硬：inspectionInfoCode 是使用者從固定分類樹上點的小類葉節點，
不是從自由文字猜的。

    uv run src/qms.py --cells            # 只掃母體（不下載），寫 cells.csv
    uv run src/qms.py --sample 3000      # 抽樣 + 下載
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
import paths  # noqa: E402

load_dotenv()

BASE = (os.getenv("QMS_API_BASE_URL") or "https://qms.example.invalid/api/qms").rstrip("/")
MODULE = "CONSTRUCTION"
PATH_CATEGORY = "qms/inspection"
TIMEOUT = httpx.Timeout(30.0, read=180.0)


class Qms:
    def __init__(self):
        self.c = httpx.Client(base_url=BASE, timeout=TIMEOUT, follow_redirects=True)
        self.token: str | None = None

    def login(self) -> None:
        emp, pw = os.getenv("QMS_EMP_ID"), os.getenv("QMS_PASSWORD")
        if not emp or not pw:
            raise RuntimeError("缺 QMS_EMP_ID / QMS_PASSWORD，請填 .env")
        r = self.c.post("/v1/auth/login/admin", json={"empId": emp, "password": pw})
        r.raise_for_status()
        self.token = r.json()["data"]["token"]

    def _call(self, method: str, path: str, **kw):
        r = self.c.request(method, path, headers={"Authorization": f"Bearer {self.token}"}, **kw)
        r.raise_for_status()
        body = r.json()
        if body.get("code") not in (0, 200, None):
            raise RuntimeError(f"{path} → {body.get('msg')}")
        return body["data"]

    def constructions(self) -> list[dict]:
        return self._call("GET", "/v1/building/construction/summary", params={"module": MODULE})

    def tree(self) -> list[dict]:
        return self._call("GET", "/v1/building/construction/inspection/info")

    def cells(self, construction_id: str, code: str) -> list[dict]:
        d = self._call("POST", "/v1/building/construction/summary/detail",
                       json={"module": MODULE, "constructionId": construction_id,
                             "inspectionInfoCode": code})
        return d.get("detail") or []

    def cell_photos(self, cell_id: str) -> dict:
        return self._call("GET", f"/v1/building/construction/inspection/{cell_id}",
                          params={"module": MODULE})

    def urls(self, file_ids: list[str]) -> dict[str, str]:
        d = self._call("POST", "/v1/file/info", params={"pathCategory": PATH_CATEGORY},
                       json={"id": file_ids})
        d = d if isinstance(d, list) else [d]
        return {f["id"]: f["url"] for f in d if f.get("url")}

    def close(self):
        self.c.close()


def leaves(tree) -> list[tuple[str, str, str, str]]:
    """→ [(大類, 中類, 小類名, 小類code)]。只有葉節點的 code 查得到格子。"""
    nm = lambda n: (n["name"]["zh-TW"] if isinstance(n["name"], dict) else n["name"]).strip()
    out = []
    for a in tree:
        for b in a.get("children") or []:
            kids = b.get("children") or []
            if not kids:
                out.append((nm(a), nm(b), nm(b), b["code"]))
            out += [(nm(a), nm(b), nm(c), c["code"]) for c in kids]
    return out


def scan_cells(log=print) -> pd.DataFrame:
    """掃母體：每個建案 × 每個小類 → 格子清單。只打 query，不下載照片。"""
    paths.ensure_dirs()
    q = Qms()
    q.login()
    constrs = [c for c in q.constructions() if (c.get("photoCount") or 0) > 0]
    lv = leaves(q.tree())
    log(f"{len(constrs)} 個建案 × {len(lv)} 個小類")

    rows = []
    for c in constrs:
        for i, (l1, l2, l3, code) in enumerate(lv, 1):
            try:
                cells = q.cells(c["id"], code)
            except Exception as e:
                log(f"  {c['name']}/{code} 失敗：{e}")
                continue
            for cell in cells:
                rows.append({
                    "constructionId": c["id"], "constrName": c["name"],
                    "l1": l1, "l2": l2, "l3": l3, "code": code,
                    "constructionInsId": cell["constructionInsId"],
                    "floor": cell.get("floor"), "room": cell.get("room"),
                    "status": cell.get("status"), "count": cell.get("count") or 0,
                })
            if i % 60 == 0:
                log(f"  {c['name']} {i}/{len(lv)}  已收 {len(rows)} 格")
    q.close()

    df = pd.DataFrame(rows)
    paths.QMS_CELLS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.QMS_CELLS, index=False)
    log(f"母體：{len(df)} 格 / {int(df['count'].sum())} 張 → {paths.QMS_CELLS}")
    return df


def sample(n: int = 3000, per_class_cap: int = 200, seed: int = 0, log=print) -> dict:
    """隨機抽 n 張下載。

    per_class_cap 是必要的煞車：防水隔熱佔全量 1/3，純隨機會讓它吃掉抽樣的三分之一，
    小類則一張都抽不到。設 0 可關掉，變成純隨機。
    """
    paths.ensure_dirs()
    if not paths.QMS_CELLS.exists():
        scan_cells(log=log)
    cells = pd.read_csv(paths.QMS_CELLS)
    cells = cells[cells["count"] > 0]

    rng = random.Random(seed)
    order = cells.sample(frac=1, random_state=seed)          # 洗牌 = 隨機
    picked, per_class, total = [], {}, 0
    for r in order.itertuples():
        if total >= n:
            break
        if per_class_cap and per_class.get(r.code, 0) >= per_class_cap:
            continue
        picked.append(r)
        per_class[r.code] = per_class.get(r.code, 0) + r.count
        total += r.count
    log(f"抽中 {len(picked)} 格 / 約 {total} 張 / {len(per_class)} 個小類")

    q = Qms()
    q.login()
    have = {p.stem for p in paths.QMS_PHOTOS.iterdir() if p.is_file()}
    rows, ok, fail = [], 0, 0

    # 一格一個 API 呼叫，序列跑 43 張/分鐘太慢；查詢與下載都開執行緒池
    def fetch_cell(r):
        try:
            return r, q.cell_photos(r.constructionInsId).get("photo") or []
        except Exception as e:
            return r, e

    with ThreadPoolExecutor(max_workers=6) as pool:
        for n, (r, photos) in enumerate(pool.map(fetch_cell, picked), 1):
            if isinstance(photos, Exception):
                log(f"  格 {r.constructionInsId} 失敗：{photos}")
                fail += 1
                continue
            for p in photos:
                if not p.get("id"):
                    continue
                rows.append({
                    "fileId": p["id"], "constructionInsId": r.constructionInsId,
                    "constructionId": r.constructionId, "constrName": r.constrName,
                    "l1": r.l1, "l2": r.l2, "l3": r.l3, "code": r.code,
                    "floor": r.floor, "room": r.room, "space": p.get("space"),
                    "cellStatus": r.status, "photoStatus": p.get("status"),
                    "displayStatus": p.get("displayStatus"), "signStatus": p.get("signStatus"),
                    "creator": p.get("creator"), "isVendorUpload": p.get("isVendorUpload"),
                    "remark": p.get("remark"),
                })
            if n % 200 == 0:
                log(f"  查詢 {n}/{len(picked)} 格")

    todo = [r["fileId"] for r in rows if r["fileId"] not in have]
    todo = list(dict.fromkeys(todo))
    log(f"需下載 {len(todo)} 張")

    def grab(fid_url):
        fid, url = fid_url
        try:
            blob = httpx.get(url, timeout=TIMEOUT, follow_redirects=True).content
            (paths.QMS_PHOTOS / f"{fid}.jpg").write_bytes(blob)
            return True
        except Exception:
            return False

    for i in range(0, len(todo), 100):          # 簽名網址一次換 100 個，1 小時內用完
        chunk = todo[i:i + 100]
        try:
            url_map = q.urls(chunk)
        except Exception as e:
            log(f"  取號失敗：{e}")
            fail += len(chunk)
            continue
        with ThreadPoolExecutor(max_workers=10) as pool:
            for good in pool.map(grab, url_map.items()):
                ok += good
                fail += not good
        fail += len(chunk) - len(url_map)
        if (i // 100) % 5 == 0:
            log(f"  下載 {ok}/{len(todo)}")
    q.close()

    df = pd.DataFrame(rows).drop_duplicates(subset="fileId")
    df.to_csv(paths.QMS_MANIFEST, index=False)
    stat = {"cells": len(picked), "photos": len(df), "downloaded": ok, "failed": fail,
            "classes": int(df.code.nunique()) if len(df) else 0}
    log(f"完成：{stat} → {paths.QMS_MANIFEST}")
    return stat


LABEL_CFG = paths.ROOT / "reference" / "qms_labels.yaml"


def labeled(min_class_size: int | None = None, completed_only: bool = False) -> pd.DataFrame:
    """QMS manifest → cls = 中類（L2），再套 reference/qms_labels.yaml 的調整。

    小類 211 個、平均 14 張，訓不動；大類 11 個又把防水/機電內部差異抹平。
    中類是分類樹本來就有的層級，也最接近日報那 10 類的顆粒度。

    exclude / merge 依混淆矩陣決定，只動訓練與評估的標籤，不動 QMS 的分類樹本身。
    """
    import yaml

    cfg = yaml.safe_load(LABEL_CFG.read_text()) if LABEL_CFG.exists() else {}
    cfg = cfg or {}
    df = pd.read_csv(paths.QMS_MANIFEST)
    if completed_only:                       # 退件重拍的照片本身可能就是拍壞的
        df = df[df.cellStatus == "COMPLETE"]
    df = df[~df.l2.isin(cfg.get("exclude") or [])]
    df = df.assign(cls=df.l2.replace(cfg.get("merge") or {}))
    n = df.cls.value_counts()
    floor = min_class_size if min_class_size is not None else int(cfg.get("min_class_size", 30))
    return df[df.cls.isin(n[n >= floor].index)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", action="store_true", help="只掃母體，不下載")
    ap.add_argument("--stats", action="store_true", help="看已下載那批的 L2 分佈")
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--per-class-cap", type=int, default=200, help="0 = 純隨機")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cells:
        scan_cells()
    elif a.stats:
        d = labeled()
        print(d.cls.value_counts().to_string())
        print(f"\n{len(d)} 張 / {d.cls.nunique()} 個中類 / {d.constructionInsId.nunique()} 格")
    else:
        sample(a.sample, a.per_class_cap, a.seed)
