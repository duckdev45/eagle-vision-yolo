"""階段 A：同步日報 + 下載照片。

鐵律（SPEC §2.1）：raw/ 抄下來就不再改。這裡只做「照抄」——
不分類、不改檔名、不過濾標籤、不正規化。分類是 labels.py 的事。

冪等：照片檔存在就跳過；manifest 以 fileId upsert。
增量：SUBMITTED 且 version 未變的日報不重抓 detail；DRAFT 一律重抓。

    uv run src/sync.py [--full] [--constr <id>] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import sys
from datetime import UTC, datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import api as api_module
import paths
from api import Pms
from photo_quality import bytes_problem, file_problem

PHOTO_KINDS = ("WORK_ITEM", "WORKFORCE")  # FREE_CONTENT 的 progressShot 無 title 語意，排除
MANIFEST_COLS = [
    "fileId",
    "source",
    "title",
    "specKey",
    "location",
    "chipsOn",
    "chipsCustom",
    "predWorkItem",
    "predPos",
    "annoRaw",
    "annoV",
    "natW",
    "natH",
    "tradeName",
    # 2026-08-17 新 prompt 多出來的欄位。前端還沒寫進 anno 之前一律 null，
    # 有就抓、沒有就空——不能等欄位上線才補程式，那幾天的資料會流失。
    # 這 7 欄 + predWorkItem/predPos 剛好蓋滿 PostDailyReportLlmWorkItemRes 的 9 個
    # required 欄位。契約沒有的欄位不要自己開，開了就是一欄永遠不會有值的 null。
    "predAction",
    "predInspPoints",
    "predBoxes",
    "predSecondary",
    "predLocation",
    "predConf",
    "promptVersion",
    "dailyReportInfoId",
    "constrId",
    "constrName",
    "reportDate",
    "status",
    "stage",
    "createdBy",
    "pageSort",
    "serial",
    "pathCategory",
    "objectKey",
    "fileName",
    "mimeType",
    "size",
    "desc",
    "remark",
    "active",
    "syncedAt",
    "apiHost",  # 這張是哪台 PMS 抄來的。切換環境時 merge_manifest 靠它判斷「消失」
]

# 2026-08-25 切正式版之前抄的照片沒有這一欄。不能預設成當前主機——那等於謊稱
# 它們是正式版來的，下次全量同步就會因為「正式版清單裡沒有」而被作廢 60 張已標註的照片。
UNKNOWN_HOST = "unknown"


def _ext(name: str | None, mime: str | None) -> str:
    if name and "." in name:
        return "." + name.rsplit(".", 1)[-1].lower()
    return mimetypes.guess_extension(mime or "") or ".bin"


def _store_valid_photo(file_id: str, ext: str, blob: bytes, old_paths: list) -> None:
    """有效下載原子替換壞檔；舊位元組先留在 raw/quarantine 供追查。"""
    dst = paths.PHOTOS / f"{file_id}{ext}"
    tmp = dst.with_name(f".{dst.name}.download")
    invalid = [p for p in old_paths if file_problem(p)]
    quarantine = paths.RAW / "quarantine"
    tmp.write_bytes(blob)
    try:
        if invalid:
            quarantine.mkdir(parents=True, exist_ok=True)
        for old in invalid:
            digest = hashlib.sha256(old.read_bytes()).hexdigest()[:12]
            archived = quarantine / f"{old.stem}-{digest}{old.suffix}"
            if old == dst:
                if not archived.exists():
                    shutil.copy2(old, archived)
        os.replace(tmp, dst)
        for old in invalid:
            if old != dst:
                digest = hashlib.sha256(old.read_bytes()).hexdigest()[:12]
                archived = quarantine / f"{old.stem}-{digest}{old.suffix}"
                if archived.exists():
                    old.unlink()
                else:
                    os.replace(old, archived)
    finally:
        tmp.unlink(missing_ok=True)


def _chips(fe_meta: dict | None) -> tuple[str, str]:
    """feMeta.pool[] → (勾選的查驗重點, 自填的查驗重點)。結構未定型，容錯處理。"""
    pool = (fe_meta or {}).get("pool") or []
    if not isinstance(pool, list):
        return "", ""
    on, custom = [], []
    for c in pool:
        if not isinstance(c, dict):
            continue
        label = c.get("label") or c.get("name") or c.get("text") or ""
        if c.get("on"):
            on.append(str(label))
        if c.get("custom"):
            custom.append(str(c.get("custom") if isinstance(c.get("custom"), str) else label))
    return "|".join(filter(None, on)), "|".join(filter(None, custom))


def _photos(content: dict) -> list[dict]:
    """一頁裡的所有照片。WORKFORCE 的出工照掛在 items[].photos 底下。"""
    out = list(content.get("photos") or [])
    for item in content.get("items") or []:
        if isinstance(item, dict):
            out += [
                {**p, "workforceTradeId": item.get("workforceTradeId")} for p in (item.get("photos") or [])
            ]
    return out


# anno 在契約裡是 additionalProperties:true 的自由 blob，「後端不解讀」——
# 前端加欄位不會有人通知我們。已知的列在這，冒出別的就喊，不然只會靜靜漏資料。
KNOWN_ANNO_KEYS = {
    "pv",
    "workItem",
    "wiConf",
    "secItems",
    "loc",
    "workAction",  # LLM 契約的 9 欄
    "inspPoints",
    "inspectionPoints",
    "evidence",
    "wmPos",
    "watermarkPosition",
    "v",
    "raw",
    "nat",  # 版本 / 是否無標註原圖 / 原尺寸：有收
    "date",
    "fit",
    "crop",
    "marks",  # 前端顯示用（裁切、縮放、標註向量）：刻意不收
}


def anno_keys(report: dict) -> set[str]:
    return {
        k
        for page in report.get("pages") or []
        for p in _photos(page.get("content") or {})
        for k in (p.get("anno") or {})
    }


def flatten(report: dict, api_host: str = "") -> list[dict]:
    """日報 JSON → 一列一張照片。純函式，不碰網路（好測）。"""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    base = {
        "dailyReportInfoId": report.get("dailyReportInfoId"),
        "constrId": report.get("constrId"),
        "constrName": (report.get("constr") or {}).get("name") or report.get("constrName"),
        "reportDate": report.get("reportDate"),
        "status": report.get("status"),
        "stage": report.get("stage"),
        "createdBy": report.get("createdBy"),  # accountId（uuid）；測試帳號過濾用
    }
    rows: list[dict] = []
    for page in report.get("pages") or []:
        kind = page.get("contentKind")
        if kind not in PHOTO_KINDS:
            continue
        content = page.get("content") or {}
        fe = content.get("feMeta") or {}
        chips_on, chips_custom = _chips(fe)
        trade_by_id = {
            i.get("workforceTradeId"): i.get("tradeName")
            for i in (content.get("items") or [])
            if isinstance(i, dict)
        }

        for p in _photos(content):
            fid = p.get("id") or p.get("fileId")
            if not fid:
                continue
            anno = p.get("anno") or {}
            nat = anno.get("nat") or {}
            # anno.pv 是後端寫的真版本；沒有的話只能靠欄位有無反推（同一份日報
            # 裡新舊混雜過，日期分界不管用），那是猜的，標成 *-guess 以示區別。
            # 沒有 workItem = 這張根本沒送過 Gemini，不是「舊版 prompt」，留空。
            pv = anno.get("pv") or (
                ("v2-guess" if anno.get("workAction") else "v1-guess") if anno.get("workItem") else None
            )
            name, mime = p.get("name"), p.get("type")
            path_cat = p.get("pathCategory") or "daily-report"
            rows.append(
                {
                    **base,
                    "fileId": fid,
                    "source": kind,
                    "title": content.get("title"),
                    "specKey": fe.get("specKey"),
                    "location": fe.get("location"),
                    "chipsOn": chips_on,
                    "chipsCustom": chips_custom,
                    "predWorkItem": anno.get("workItem"),
                    "predPos": anno.get("wmPos") or anno.get("watermarkPosition"),
                    "predAction": anno.get("workAction"),
                    # 實際存的鍵是 inspPoints（不是 API 的 inspectionPoints），兩種都收
                    "predInspPoints": json.dumps(
                        anno.get("inspPoints") or anno.get("inspectionPoints"), ensure_ascii=False
                    )
                    if (anno.get("inspPoints") or anno.get("inspectionPoints"))
                    else None,
                    # 前端把 evidenceTargets 壓成 anno.evidence 的 {label, box, conf}
                    "predBoxes": json.dumps(anno.get("evidence"), ensure_ascii=False)
                    if anno.get("evidence")
                    else None,
                    "predSecondary": json.dumps(anno.get("secItems"), ensure_ascii=False)
                    if anno.get("secItems")
                    else None,
                    "predLocation": anno.get("loc"),
                    "predConf": anno.get("wiConf"),
                    "promptVersion": pv,
                    "annoRaw": anno.get("raw"),
                    "annoV": anno.get("v"),
                    "natW": nat.get("w") if isinstance(nat, dict) else None,
                    "natH": nat.get("h") if isinstance(nat, dict) else None,
                    "tradeName": trade_by_id.get(p.get("workforceTradeId")),
                    "pageSort": page.get("pageSort"),
                    "serial": p.get("serial") or p.get("sortOrder"),
                    "pathCategory": path_cat,
                    "objectKey": f"{path_cat}/{fid}{_ext(name, mime)}",
                    "fileName": name,
                    "mimeType": mime,
                    "size": p.get("size"),
                    "desc": p.get("desc"),
                    "remark": p.get("remark"),
                    "_url": p.get("url"),
                    "active": True,
                    "syncedAt": now,
                    "apiHost": api_host,
                }
            )
    return rows


def _load(path, cols) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return pd.DataFrame(columns=cols)


def _truthy(s: pd.Series) -> pd.Series:
    """CSV 讀回來的 active 是字串 "False"，bool("False") 是 True——不能直接 astype。"""
    return s.astype(str).str.lower().isin(["true", "1"])


def merge_manifest(old: pd.DataFrame, new: pd.DataFrame, partial: bool, api_host: str = "") -> pd.DataFrame:
    """upsert：以最新 metadata 為準。

    只有全量跑才知道某張真的從 PMS 消失了。--constr/--limit 只看了母體的一片，
    把沒列到的標成 active=false 會一次作廢掉其餘全部（跑一次 --limit 5 就毀了）。

    「消失」只在**同一台主機**內成立。2026-08-25 從 pms-dev 切到 pms，兩邊
    dailyReportInfoId 與 fileId 完全相同，但正式版只有 8/01 之後的日報——
    照舊邏輯跑一次全量，dev 那 81 張（其中 60 張已標註）會集體被標成 active=false，
    訓練集無聲少掉 9%。它們沒有從 PMS 消失，是我們換了台機器問。
    """
    if not len(old):
        return new
    kept = old[~old.fileId.isin(new.fileId)].copy()
    if not partial:
        if "apiHost" not in kept:
            kept["apiHost"] = UNKNOWN_HOST
        # 只作廢同一台主機的。別台與 unknown（切換前抄的）一律留著
        same = kept.apiHost.fillna(UNKNOWN_HOST) == (api_host or UNKNOWN_HOST)
        # CSV 讀回來 active 是 str dtype，.loc 部分賦值塞 bool 會 TypeError
        # （pandas 3 的 str dtype 不吃 bool）。先轉 object 再改那幾列。
        kept["active"] = kept["active"].astype(object)
        kept.loc[same, "active"] = False
    return pd.concat([new, kept], ignore_index=True)


def merge_index(old: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """水位線同樣要 upsert：整檔覆寫成子集的話，--limit 跑一次就把其餘日報的
    version 清光，下次全量會把 100+ 份 detail 全部重抓一遍。"""
    if not len(fresh):
        return old
    if not len(old):
        return fresh
    return pd.concat([fresh, old[~old.dailyReportInfoId.isin(fresh.dailyReportInfoId)]], ignore_index=True)


def sync(full: bool = False, constr_id: str | None = None, limit: int | None = None, log=print) -> dict:
    paths.ensure_dirs()
    partial = bool(constr_id or limit)  # 只看了母體的一片，不足以判定「消失」
    pms = Pms()
    api_host = api_module.host()
    log(f"來源 {api_host}")
    pms.login()

    index = _load(paths.INDEX, ["dailyReportInfoId", "version", "status", "fetchedAt"])
    seen_version = (
        {r.dailyReportInfoId: (r.status, str(r.version)) for r in index.itertuples()} if len(index) else {}
    )

    listed = list(
        pms.list_reports(
            constr_id=constr_id,
            start_date=os.getenv("SYNC_START_DATE") or None,
            end_date=os.getenv("SYNC_END_DATE") or None,
        )
    )
    # 靜靜少抓一頁是最難發現的錯，拿伺服器的 dataCnt 對帳
    if pms.last_total is not None and len(listed) != pms.last_total:
        log(f"  ⚠ 清單 {len(listed)} 份 ≠ 伺服器 dataCnt {pms.last_total}，分頁可能沒走完")
    if limit:
        listed = listed[:limit]
    log(f"日報清單 {len(listed)} 份")

    rows: list[dict] = []
    seen_keys: set[str] = set()
    fetched = 0
    for i, meta in enumerate(listed, 1):
        rid = meta["dailyReportInfoId"]
        cached = paths.REPORTS_JSON / f"{rid}.json"
        prev = seen_version.get(rid)
        unchanged = (
            not full
            and cached.exists()
            and prev is not None
            and prev == ("SUBMITTED", str(meta.get("version")))
            and meta.get("status") == "SUBMITTED"
        )
        if unchanged:
            report = json.loads(cached.read_text())
        else:
            report = pms.get_report(rid)
            cached.write_text(json.dumps(report, ensure_ascii=False))
            fetched += 1
        report.setdefault("constrName", meta.get("constrName"))
        rows.extend(flatten(report, api_host))
        seen_keys |= anno_keys(report)
        if i % 20 == 0:
            log(f"  ...{i}/{len(listed)}")

    if seen_keys - KNOWN_ANNO_KEYS:
        log(
            f"  ⚠ anno 出現沒收的欄位：{sorted(seen_keys - KNOWN_ANNO_KEYS)}"
            f" —— 對一下 api-docs-json，該收的補進 flatten()"
        )

    new_df = pd.DataFrame(rows, columns=[*MANIFEST_COLS, "_url"])
    urls = dict(zip(new_df.fileId, new_df._url)) if len(new_df) else {}
    new_df = new_df.drop(columns=["_url"]).drop_duplicates(subset="fileId", keep="last")

    # reindex 才會真的丟掉已刪除的欄位——舊 CSV 的欄位會跟著 kept 那半邊 concat 回來
    merged = merge_manifest(_load(paths.MANIFEST, MANIFEST_COLS), new_df, partial, api_host)
    merged = merged.reindex(columns=MANIFEST_COLS)
    merged.to_csv(paths.MANIFEST, index=False)

    # 下載尚未存在的照片。太小的檔是伺服器端壞檔（實測有 19 bytes 的），
    # 只看檔名存在的話它永遠算「已有」，一輩子不會補抓。
    local = {}
    for photo in paths.PHOTOS.iterdir():
        if photo.is_file() and not photo.name.startswith("."):
            local.setdefault(photo.stem, []).append(photo)
    have = {fid for fid, photos in local.items() if any(file_problem(p) is None for p in photos)}
    todo = [r for r in new_df.itertuples() if r.fileId not in have]
    ok = fail = 0
    for n, r in enumerate(todo, 1):
        # 簽名網址只有 1 小時，快取 JSON 裡的早就過期（403）→ 失敗一次就重新取號再試
        blob = err = None
        for attempt in (0, 1):
            url = (
                urls.get(r.fileId)
                if attempt == 0
                else pms.file_urls([r.fileId], r.pathCategory).get(r.fileId)
            )
            if not url:
                continue
            try:
                blob = pms.download(url)
                break
            except Exception as e:  # 單張失敗不中斷整批
                err = e
        if blob is None:
            log(f"  下載失敗 {r.fileId}: {err or '取不到網址'}")
            fail += 1
            continue
        if problem := bytes_problem(blob):  # 壞檔不落地，下次還會再試
            log(f"  壞檔 {r.fileId}: {problem}")
            fail += 1
            continue
        try:
            _store_valid_photo(r.fileId, _ext(r.fileName, r.mimeType), blob, local.get(r.fileId, []))
        except OSError as exc:
            log(f"  儲存失敗 {r.fileId}: {exc}")
            fail += 1
            continue
        ok += 1
        if n % 50 == 0:
            log(f"  下載 {n}/{len(todo)}")

    fresh = pd.DataFrame(
        [
            {
                "dailyReportInfoId": m["dailyReportInfoId"],
                "version": m.get("version"),
                "status": m.get("status"),
                "fetchedAt": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            for m in listed
        ],
        columns=list(index.columns),
    )
    merge_index(index, fresh).to_csv(paths.INDEX, index=False)
    pms.close()

    stat = {
        "reports": len(listed),
        "reportsFetched": fetched,
        "photos": len(merged),
        "downloaded": ok,
        "failed": fail,
        "inactive": int((~_truthy(merged.active)).sum()),
    }
    log(f"完成：{stat}")
    return stat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="忽略快取，全部重抓 detail")
    ap.add_argument("--constr", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    sync(full=a.full, constr_id=a.constr, limit=a.limit)
