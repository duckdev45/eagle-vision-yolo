"""PMS 工種工作台命令列。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import pms_exchange as exchange
from core import pms_review as review
from core import pms_store as store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest="command", required=True)
    subs.add_parser("status")
    subs.add_parser("candidates")
    subs.add_parser("calibration")
    export = subs.add_parser("export")
    export.add_argument("--ids", help="逗號分隔的 PMS fileId；未提供則取待複核照片")
    export.add_argument("--limit", type=int, default=12)
    export.add_argument("--out", type=Path, required=True)
    imp = subs.add_parser("import")
    imp.add_argument("--input", type=Path, required=True)
    ai = subs.add_parser("ai", help="OpenAI 看圖與標題自動產生分類建議")
    ai.add_argument("--ids", help="逗號分隔的 WORK_ITEM 照片識別碼")
    ai.add_argument("--limit", type=int, default=4)
    ai.add_argument("--model", default="", help="省略時使用 PMS_OPENAI_MODEL")
    ai.add_argument("--force", action="store_true", help="重新呼叫 API，不沿用同內容的結果")
    decision = subs.add_parser("decide")
    decision.add_argument("--id", required=True)
    decision.add_argument("--action", choices=["classified", "uncertain", "excluded"], required=True)
    decision.add_argument("--label", default="")
    decision.add_argument("--reviewer", required=True)
    decision.add_argument("--reason", default="")
    decision.add_argument("--proposal", default="")
    train = subs.add_parser("train")
    train.add_argument("--name", required=True)
    train.add_argument("--sync", action="store_true")
    args = ap.parse_args(argv)
    try:
        result = None
        if args.command == "status":
            df, model = review.snapshot()
            result = {
                "source": "pms",
                "photoSource": "WORK_ITEM",
                "photos": len(df),
                "model": model["name"],
                "catalogClasses": len(review.catalog()),
                "modelClasses": len(model["classes"]),
                "states": df.reviewState.value_counts().to_dict(),
                "routes": df.route.value_counts().to_dict(),
                "needsReview": int(df.needsReview.sum()),
                "warning": model["warning"],
            }
        elif args.command == "calibration":
            result = review.calibration()
        elif args.command == "candidates":
            df, _ = review.snapshot()
            result = {
                "registered": list(store.latest("candidate").values()),
                "discovery": review.discover(df),
            }
        elif args.command == "export":
            if args.out.exists():
                raise ValueError("輸出檔已存在，請使用新檔名。")
            if not 1 <= args.limit <= 50:
                raise ValueError("limit 必須介於 1 與 50。")
            if args.ids:
                ids = [fid.strip() for fid in args.ids.split(",") if fid.strip()]
            else:
                df, _ = review.snapshot()
                ids = (
                    df[df.needsReview.astype(bool)]
                    .sort_values("reportDate", ascending=False)
                    .head(args.limit)
                    .fileId.tolist()
                )
            packet, data = exchange.export_packet(ids)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("xb") as fh:
                fh.write(data)
            result = {"packetId": packet, "file": str(args.out), "photos": len(set(ids))}
        elif args.command == "import":
            result = {
                "importedSuggestions": exchange.import_suggestions(
                    json.loads(args.input.read_text(encoding="utf-8"))
                )
            }
        elif args.command == "decide":
            review.decide(
                args.id,
                args.action,
                label=args.label,
                reviewer=args.reviewer,
                reason=args.reason,
                proposal_id=args.proposal,
            )
            result = {"fileId": args.id, "action": args.action}
        elif args.command == "ai":
            from core import pms_vision

            if not 1 <= args.limit <= pms_vision.MAX_BATCH:
                raise ValueError(f"limit 必須介於 1 與 {pms_vision.MAX_BATCH}。")
            if args.ids:
                ids = [fid.strip() for fid in args.ids.split(",") if fid.strip()]
            else:
                df, _ = review.snapshot()
                ids = (
                    df[df.needsReview.astype(bool)]
                    .sort_values("reportDate", ascending=False)
                    .head(args.limit)
                    .fileId.tolist()
                )
            result = pms_vision.classify(ids, model=args.model, force=args.force)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result["failed"] else 0
        elif args.command == "train":
            from pipeline.pms_workflow import run

            run(args.name, with_data=args.sync)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=int))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
