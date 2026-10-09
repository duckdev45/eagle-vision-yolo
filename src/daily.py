"""每日編排（無人值守）：同步 → 前處理／特徵 → 分流 → 夠多新資料就重訓 → 考卷過關自動切換。

    uv run --extra train src/daily.py              # 排程跑的就是這個（make daily）
    uv run --extra train src/daily.py --dry-run    # 只算分流、印出「會不會重訓」，不同步不訓練
    uv run src/daily.py --only-route               # 只重算分流（不需 train extra）

每一步失敗都不中斷後面能做的事：同步失敗（網路、帳密）照樣用本機資料分流；重訓失敗不影響
今天的收件匣。結果附加到 data/daily-log.jsonl，最後一行印摘要（排程把它當通知內容）。

重訓觸發（任一）：
    · 現行模型之後新增 ≥ RETRAIN_NEW_REVIEWS 筆人審（人教了新東西，越早學到越好）
    · 距現行模型訓練 ≥ RETRAIN_DAYS 天，且有 ≥ RETRAIN_NEW_PHOTOS 張模型沒看過的新照片
切換由 core.promotion 的公平考卷決定（2026-10-09 定案：過關就自動換）。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import paths

RETRAIN_NEW_REVIEWS = 20
RETRAIN_NEW_PHOTOS = 50
RETRAIN_DAYS = 7


def should_retrain(new_reviews: int, new_photos: int, days_since: float) -> tuple[bool, str]:
    """純函式：要不要重訓、為什麼（理由寫進每日紀錄）。"""
    if new_reviews >= RETRAIN_NEW_REVIEWS:
        return True, f"新增 {new_reviews} 筆人審（≥ {RETRAIN_NEW_REVIEWS}）"
    if days_since >= RETRAIN_DAYS and new_photos >= RETRAIN_NEW_PHOTOS:
        return True, f"距上次訓練 {days_since:.0f} 天、{new_photos} 張新照片"
    return False, f"人審 +{new_reviews}、新照片 {new_photos}、距上次訓練 {days_since:.0f} 天，未達門檻"


def training_debt() -> dict:
    """現行模型訓練之後，累積了多少它沒學過的東西。"""
    import split as split_mod

    name = split_mod.current()
    probe = split_mod.probe_path(name)
    trained = (
        datetime.fromtimestamp(probe.stat().st_mtime, UTC)
        if probe.exists()
        else datetime.min.replace(tzinfo=UTC)
    )
    reviews = 0
    if paths.REVIEW.exists():
        df = pd.read_csv(paths.REVIEW, dtype=str, keep_default_na=False)
        at = pd.to_datetime(df.reviewedAt, errors="coerce", utc=True)
        reviews = int((at > trained).sum())
    route = paths.ROUTE / "latest.json"
    new_photos = (
        json.loads(route.read_text(encoding="utf-8")).get("signals", {}).get("live", 0)
        if route.exists()
        else 0
    )
    days = (datetime.now(UTC) - trained).total_seconds() / 86400
    return {"model": name, "newReviews": reviews, "newPhotos": int(new_photos), "daysSince": round(days, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="只分流與評估是否重訓，不同步、不訓練、不切換")
    ap.add_argument("--only-route", action="store_true", help="只重算分流")
    ap.add_argument("--no-sync", action="store_true", help="跳過同步（離線時）")
    args = ap.parse_args(argv)
    lines: list[str] = []

    def log(*parts) -> None:
        msg = " ".join(str(p) for p in parts)
        lines.append(msg)
        print(msg, flush=True)

    record: dict = {"startedAt": datetime.now(UTC).isoformat(timespec="seconds"), "steps": {}}

    def step(key: str, fn) -> Any:
        try:
            out = fn()
            record["steps"][key] = "ok"
            return out
        except (Exception, SystemExit) as exc:  # 一步壞了，後面能做的照做
            record["steps"][key] = f"failed: {type(exc).__name__}: {exc}"
            log(f"⚠ {key} 失敗：{exc}")
            traceback.print_exc(file=sys.stderr)
            return None

    if not (args.dry_run or args.only_route):
        import features
        import prepare
        import sync

        if not args.no_sync:
            step("sync", lambda: sync.sync(log=log))
        step("prepare", lambda: prepare.run(kind="report", log=log))
        for key in dict.fromkeys((features.DEFAULT_ENCODER, features.LEGACY_ENCODER)):
            step(f"features-{key}", lambda key=key: features.extract(model_key=key, src="report", log=log))

    from core import routing

    summary = step("route", lambda: routing.build(log=log))
    if summary:
        record["route"] = {
            "model": summary["model"],
            "counts": summary["counts"],
            "audit": summary["auditPrecision"],
        }

    if not args.only_route:
        debt = step("debt", training_debt) or {}
        record["debt"] = debt
        go, why = should_retrain(
            debt.get("newReviews", 0), debt.get("newPhotos", 0), debt.get("daysSince", 0)
        )
        record["retrain"] = {"go": go, "why": why}
        log(("🔁 重訓：" if go else "⏭ 不重訓：") + why)
        if go and not args.dry_run:
            from core import promotion
            from pipeline import pms_workflow

            name = pms_workflow.next_version()
            record["retrain"]["name"] = name
            trained = step("train", lambda: pms_workflow.run(name, log=log, with_explain=False) or True)
            if trained:
                result = step("promote", lambda: promotion.promote(name, log=log))
                if result:
                    record["promotion"] = {
                        k: result.get(k) for k in ("candidate", "baseline", "promoted", "reasons")
                    }
                    if result.get("promoted"):
                        step("route-after-promote", lambda: routing.build(log=log))  # 收件匣改用新模型的判斷

    record["finishedAt"] = datetime.now(UTC).isoformat(timespec="seconds")
    if not args.dry_run:
        paths.DAILY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with paths.DAILY_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    failed = [k for k, v in record["steps"].items() if v != "ok"]
    counts = (record.get("route") or {}).get("counts", {})
    print(
        "DAILY "
        + json.dumps(
            {
                "收件匣": counts.get("queue", 0) + counts.get("audit", 0),
                "自動確認": counts.get("auto", 0),
                "隔離": counts.get("quarantine", 0),
                "重訓": record.get("retrain", {}).get("why", ""),
                "切換": (record.get("promotion") or {}).get("promoted"),
                "失敗步驟": failed,
            },
            ensure_ascii=False,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
