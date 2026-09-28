"""本機唯讀影子推論 API，不寫 PMS 日報或人工答案。

    uv run --extra train python src/pms_shadow_api.py --model v40
    curl -s http://127.0.0.1:8765/v1/pms/photo-predictions \
      -H 'Content-Type: application/json' -d '{"fileId":"..."}'

僅綁定 loopback；正式跨主機串接須由 PMS 後端加入身分驗證、授權及部署控管。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pms_inference import InferenceError, PmsShadowPredictor


def handler_for(predictor: PmsShadowPredictor):
    class ShadowHandler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._json(200, {"status": "ready", "modelVersion": predictor.version})
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/v1/pms/photo-predictions":
                self._json(404, {"error": "not_found"})
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self._json(415, {"error": "json_required"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if size <= 0 or size > 2048:
                self._json(400, {"error": "invalid_request_size"})
                return
            try:
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict) or set(request) != {"fileId"}:
                    raise ValueError("請只提交 fileId。")
                file_id = request["fileId"]
                if not isinstance(file_id, str) or not file_id:
                    raise ValueError("fileId 必須是非空字串。")
            except (ValueError, UnicodeDecodeError) as exc:
                self._json(400, {"error": "invalid_request", "detail": str(exc)})
                return
            start = time.monotonic()
            try:
                result = predictor.predict(file_id)
            except InferenceError as exc:
                status = 404 if exc.code == "photo_unavailable" else 422
                self._json(status, {"error": exc.code, "detail": str(exc)})
                self._audit(file_id, exc.code, start)
            except ValueError as exc:
                self._json(400, {"error": "invalid_file_id", "detail": str(exc)})
                self._audit(file_id, "invalid_file_id", start)
            except Exception:
                self._json(500, {"error": "inference_failed"})
                self._audit(file_id, "inference_failed", start)
            else:
                self._json(200, result)
                self._audit(file_id, "review_required", start)

        def _audit(self, file_id: str, status: str, start: float) -> None:
            print(
                json.dumps(
                    {
                        "fileId": file_id,
                        "modelVersion": predictor.version,
                        "status": status,
                        "latencyMs": round((time.monotonic() - start) * 1000),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return ShadowHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="PMS 本機唯讀影子推論")
    parser.add_argument("--model", default="v40")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = parser.parse_args()
    predictor = PmsShadowPredictor(args.model, args.device)
    predictor.warmup()
    print(f"PMS 影子推論 {args.model} listening on 127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), handler_for(predictor)).serve_forever()


if __name__ == "__main__":
    main()
