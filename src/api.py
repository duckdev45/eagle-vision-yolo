"""PMS REST client。

契約來源與前端 orval 同一份（URL 在 .env：API_BASE_URL）
所有回應皆為 { code, msg, data } 信封，unwrap 後才回傳。

2026-08-25 起預設指向正式版。dev 是 prod 的複本，**dailyReportInfoId 與 fileId
兩邊完全相同**（實測抽樣 12 份 173 張，重疊 173/173），所以切換主機不會重下照片、
也不會讓既有標籤失聯。差別在母體：prod 只有 2026-08-01 之後的日報，
dev 另外留著 7/23 起的 81 張（含 Project A、發大財 這種測試資料）。
manifest 因此多一欄 `apiHost` 記來源——見 sync.merge_manifest。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE = os.getenv("API_BASE_URL", "").rstrip("/")  # 由 .env 提供；見 .env.example
TIMEOUT = httpx.Timeout(30.0, read=120.0)


def host() -> str:
    """API 主機名。manifest 用它標照片是哪台來的（換主機不等於照片消失）。"""
    return BASE.split("://", 1)[-1].split("/", 1)[0]


class ApiError(RuntimeError):
    pass


class Pms:
    def __init__(self, emp_id: str | None = None, password: str | None = None):
        self.emp_id = emp_id or os.getenv("PMS_EMP_ID") or ""
        self.password = password or os.getenv("PMS_PASSWORD") or ""
        self.c = httpx.Client(base_url=BASE, timeout=TIMEOUT, follow_redirects=True)
        self.token: str | None = None
        self.last_total: int | None = None  # 上一次 list_reports 的 dataCnt，給呼叫端對帳

    # --- 內部 ---------------------------------------------------------
    def _unwrap(self, r: httpx.Response) -> Any:
        if r.status_code == 401 and self.token:
            self.login()
            r = self.c.send(
                self.c.build_request(
                    r.request.method, r.request.url, content=r.request.content, headers=self._headers()
                )
            )
        if r.status_code >= 400:
            raise ApiError(f"{r.request.method} {r.request.url} -> {r.status_code} {r.text[:300]}")
        body = r.json()
        if isinstance(body, dict) and "data" in body:
            if body.get("code") not in (0, 200, None):
                raise ApiError(f"{r.request.url} code={body.get('code')} msg={body.get('msg')}")
            return body["data"]
        return body

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _post(self, path: str, json: Any = None, params: Any = None) -> Any:
        return self._unwrap(self.c.post(path, json=json, params=params, headers=self._headers()))

    def _get(self, path: str, params: Any = None) -> Any:
        return self._unwrap(self.c.get(path, params=params, headers=self._headers()))

    # --- 端點 ---------------------------------------------------------
    def login(self) -> None:
        if not self.emp_id or not self.password:
            raise ApiError("缺 PMS_EMP_ID / PMS_PASSWORD，請填 .env")
        self.token = None
        data = self._post(
            "/v1/auth/login/password",
            {
                "empId": self.emp_id,
                "password": self.password,
                "clientType": "WEB",
            },
        )
        self.token = data["accessToken"]

    def list_reports(
        self,
        *,
        category: str = "SITE",
        constr_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        page_size: int = 100,
    ) -> Iterator[dict]:
        """分頁走完 /v1/daily-report/reports/query。

        實測回應：pageStart=目前頁、pageSize=**總頁數**、dataCnt=總筆數。
        pageSize 這名字看起來像「每頁幾筆」，其實不是，改動前先確認。
        """
        page = 1
        while True:
            body: dict[str, Any] = {
                "pageStart": page,
                "pageLimits": page_size,
                "category": category,
                "sortOrder": [{"item": "reportDate", "order": "ASC"}],
            }
            if constr_id:
                body["constrId"] = constr_id
            if start_date:
                body["startDate"] = start_date
            if end_date:
                body["endDate"] = end_date
            data = self._post("/v1/daily-report/reports/query", body)
            self.last_total = data.get("dataCnt")
            yield from data["list"]
            if page >= (data.get("pageSize") or 1) or not data["list"]:
                return
            page += 1

    def get_report(self, report_id: str) -> dict:
        return self._get(f"/v1/daily-report/reports/{report_id}")

    def file_urls(self, file_ids: list[str], path_category: str) -> dict[str, str]:
        """重新取簽名網址（照片 url 只有 1 小時；補抓時用）。"""
        data = self._post("/v1/file/info/query", {"id": file_ids}, params={"pathCategory": path_category})
        return {f["id"]: f["url"] for f in data if f.get("url")}

    def download(self, url: str) -> bytes:
        r = httpx.get(url, timeout=TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        return r.content

    def close(self) -> None:
        self.c.close()


def client() -> Pms:
    p = Pms()
    p.login()
    return p
