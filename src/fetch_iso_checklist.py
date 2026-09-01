#!/usr/bin/env python3
"""從 QMS 爬取 ISO 標準 checklist，轉成 TSV。"""
from __future__ import annotations

import httpx
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BASE = os.getenv("QMS_API_BASE_URL", "").rstrip("/")  # 由 .env 提供；見 .env.example
TIMEOUT = 30

RAW_DIR = Path(__file__).parent.parent / "reference" / "iso" / "raw"


class QmsIsoCrawler:
    def __init__(self, emp_id: str, password: str):
        self.c = httpx.Client(base_url=BASE, timeout=TIMEOUT)
        self.emp_id = emp_id
        self.password = password
        self.token: str | None = None

    def login(self) -> None:
        r = self.c.post("/v1/auth/login/admin", json={"empId": self.emp_id, "password": self.password})
        r.raise_for_status()
        self.token = r.json()["data"]["token"]
        print(f"✓ 登入成功")

    def _call(self, method: str, path: str, **kw):
        headers = {"Authorization": f"Bearer {self.token}"}
        r = self.c.request(method, path, headers=headers, **kw)
        r.raise_for_status()
        return r.json().get("data", r.json())

    def get_iso_list(self) -> list[dict]:
        """取得所有 ISO 標準大項。"""
        data = self._call("GET", "/v1/building/iso/info")
        return data if isinstance(data, list) else [data]

    def get_iso_detail(self, iso_id: str) -> dict:
        """取得單一 ISO 標準的細項（checklist）。"""
        # 嘗試多種端點格式
        for endpoint in [f"/v1/building/iso/detail/{iso_id}", f"/v1/building/iso/{iso_id}"]:
            try:
                return self._call("GET", endpoint)
            except Exception:
                continue
        raise RuntimeError(f"無法取得 {iso_id} 的細項")

    def close(self):
        self.c.close()


def parse_checklist_tree(items: list) -> list[tuple[str, str, str]]:
    """
    將 QMS checklist 轉成 [(itemNo, status, name), ...].

    status: "O" = OPTIONAL (階段節點) / "R" = REQUIRED (檢查項)
    API 返回格式：[{docNo, status, name, ...}, ...]
    """
    result = []
    for item in items:
        # API 用 docNo 而非 itemNo，status 用 OPTIONAL/REQUIRED
        item_no = item.get("docNo", "")
        status_raw = item.get("status", "")
        name = item.get("name", "")

        # 篩選無效項
        if not item_no or not status_raw:
            continue

        # 轉換 OPTIONAL/REQUIRED 為 O/R
        status = "O" if status_raw == "OPTIONAL" else "R" if status_raw == "REQUIRED" else ""
        if not status:
            continue

        result.append((item_no, status, name))

    return result


def write_tsv(doc_list: list[dict], output_file: Path) -> None:
    """將 ISO 標準寫成一份 TSV 檔。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")

    with open(output_file, "w", encoding="utf-8") as f:
        # 寫檔頭
        f.write(f"# QS 檢查項原始資料\n")
        f.write(f"# 格式: itemNo <TAB> status(O=OPTIONAL階段節點 / R=REQUIRED檢查項) <TAB> name\n")
        f.write(f"# 快照 {date_str}，取自 QMS API。name 欄位逐字照抄（含原文錯字與空格）。\n")
        f.write(f"\n")

        for doc in doc_list:
            doc_no = doc.get("docNo", "")
            iso_id = doc.get("id", doc.get("isoInfoId", ""))
            name = doc.get("name", "")
            items = doc.get("items", [])

            if not doc_no:
                continue

            # 寫 #DOC 標頭
            f.write(f"#DOC\t{doc_no}\t{iso_id}\t{name}\n")

            # 寫檢查項
            for item_no, status, item_name in items:
                f.write(f"{item_no}\t{status}\t{item_name}\n")

    print(f"✓ 寫入 {len(doc_list)} 份標準到 {output_file}")


def main(emp_id: str, password: str, output_dir: Path | None = None):
    if output_dir is None:
        output_dir = RAW_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    crawler = QmsIsoCrawler(emp_id, password)

    try:
        crawler.login()

        # 取得所有 ISO 標準
        print("取得 ISO 標準清單...")
        iso_list = crawler.get_iso_list()
        print(f"✓ 共 {len(iso_list)} 份標準")

        # 按大類分組
        by_category = defaultdict(list)
        for iso in iso_list:
            doc_no = iso.get("docNo", "")
            category = doc_no[:4] if doc_no else "XX"  # QS01, QS02, ...
            by_category[category].append(iso)

        # 逐大類取細項 + 寫檔
        for category in sorted(by_category.keys()):
            docs = by_category[category]
            print(f"\n取得 {category} ({len(docs)} 份)...")

            for doc in docs:
                doc_no = doc.get("docNo", "")
                iso_id = doc.get("id", doc.get("isoInfoId", ""))

                if not iso_id:
                    print(f"  ⚠ {doc_no}: 無 id，跳過")
                    continue

                try:
                    detail = crawler.get_iso_detail(iso_id)
                    # detail 可能是列表或包含 items 的字典
                    if isinstance(detail, list):
                        items = detail
                    elif isinstance(detail, dict):
                        items = detail.get("items", [])
                    else:
                        items = []
                    parsed = parse_checklist_tree(items)
                    doc["items"] = parsed
                    print(f"  ✓ {doc_no}: {len(parsed)} 項")
                except Exception as e:
                    print(f"  ✗ {doc_no}: {e}")

            # 寫這一大類的 TSV
            output_file = output_dir / f"qs{category[2:4]}.tsv"
            write_tsv(docs, output_file)

    finally:
        crawler.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python fetch_iso_checklist.py <emp_id> <password>")
        print("例: python fetch_iso_checklist.py 11409001 <EMP_ID>")
        sys.exit(1)

    emp_id = sys.argv[1]
    password = sys.argv[2]

    main(emp_id, password)
    print("\n✓ 完成！執行 'make qs-phases' 重建 phases.yaml")
