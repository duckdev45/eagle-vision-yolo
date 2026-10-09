"""盤點 PMS manifest、raw 與前處理照片的品質；只寫報告，不改原圖。

uv run src/pms_quality.py
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from core import paths
from photo_quality import file_problem

OUT = paths.REPORTS_OUT / "pms-photo-quality.csv"
SUMMARY = paths.REPORTS_OUT / "pms-photo-quality-summary.json"


def audit() -> tuple[pd.DataFrame, dict]:
    if not paths.MANIFEST.exists():
        raise FileNotFoundError(f"找不到 {paths.MANIFEST}")
    from labels import labeled_manifest

    manifest = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    if not {"fileId", "source", "active"} <= set(manifest.columns):
        raise ValueError("PMS manifest 需要 fileId、source、active 欄位")
    train_ids = set(labeled_manifest().fileId)
    rows = []
    for r in manifest.drop_duplicates("fileId", keep="last").itertuples(index=False):
        if not r.fileId:
            continue
        raw = sorted(p for p in paths.PHOTOS.glob(f"{r.fileId}.*") if p.is_file())
        problems = [file_problem(p) for p in raw]
        raw_problem = "missing" if not raw else ("" if None in problems else "|".join(sorted(set(problems))))
        prepared = paths.IMAGES / f"{r.fileId}.jpg"
        prepared_problem = file_problem(prepared) if not raw_problem else None
        if raw_problem or prepared_problem:
            rows.append(
                {
                    "fileId": r.fileId,
                    "source": r.source,
                    "active": r.active,
                    "reportDate": getattr(r, "reportDate", ""),
                    "apiHost": getattr(r, "apiHost", ""),
                    "trainingRelevant": r.fileId in train_ids,
                    "rawProblem": raw_problem or "",
                    "preparedProblem": prepared_problem or "",
                    "rawPaths": "|".join(str(p) for p in raw),
                }
            )
    issues = pd.DataFrame(
        rows,
        columns=[
            "fileId",
            "source",
            "active",
            "reportDate",
            "apiHost",
            "trainingRelevant",
            "rawProblem",
            "preparedProblem",
            "rawPaths",
        ],
    )
    summary = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "manifestPhotos": int(manifest.fileId.nunique()),
        "issuePhotos": len(issues),
        "trainingRelevantIssues": int(issues.trainingRelevant.sum()) if len(issues) else 0,
        "rawProblems": issues.rawProblem[issues.rawProblem != ""].value_counts().to_dict(),
        "preparedProblems": issues.preparedProblem[issues.preparedProblem != ""].value_counts().to_dict(),
    }
    return issues, summary


def main() -> int:
    issues, summary = audit()
    paths.REPORTS_OUT.mkdir(parents=True, exist_ok=True)
    issues.to_csv(OUT, index=False)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明細 → {OUT}")
    return 0 if not summary["trainingRelevantIssues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
