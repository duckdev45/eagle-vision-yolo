"""對跑中的 vision API 量延遲：單張 /v1/predict 與同工項 2 張 /v1/predict-batch。只用標準庫。

用法：python3 service/bench.py http://127.0.0.1:18000 TOKEN photo1.jpg photo2.jpg [N]
"""

from __future__ import annotations

import base64
import json
import statistics
import sys
import time
import urllib.request


def post(url: str, token: str, path: str, data: bytes, headers: dict) -> tuple[float, dict]:
    request = urllib.request.Request(url + path, data=data, headers={"X-Service-Token": token, **headers})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    return time.perf_counter() - start, body


def main(url: str, token: str, photo1: str, photo2: str, n: int = 10) -> None:
    b1, b2 = open(photo1, "rb").read(), open(photo2, "rb").read()
    single = {"X-File-Id": "bench-1", "X-Source-Policy": "clean-raw-v1", "Content-Type": "image/jpeg"}
    post(url, token, "/v1/predict", b1, single)  # 暖機
    times = sorted(post(url, token, "/v1/predict", b1, single)[0] for _ in range(n))
    batch = json.dumps(
        {
            "workItemId": "bench",
            "sourcePolicy": "clean-raw-v1",
            "photos": [
                {
                    "fileId": f"bench-{i}",
                    "contentType": "image/jpeg",
                    "dataBase64": base64.b64encode(b).decode(),
                }
                for i, b in enumerate((b1, b2), 1)
            ],
        }
    ).encode()
    batch_times = [
        post(url, token, "/v1/predict-batch", batch, {"Content-Type": "application/json"})[0]
        for _ in range(max(3, n // 2))
    ]
    print(
        f"/v1/predict 單張：中位數 {statistics.median(times) * 1000:.0f} ms，"
        f"p90 {times[int(0.9 * len(times)) - 1] * 1000:.0f} ms（n={n}）"
    )
    print(
        f"/v1/predict-batch 2 張：中位數 {statistics.median(batch_times) * 1000:.0f} ms（n={len(batch_times)}）"
    )


if __name__ == "__main__":
    main(*sys.argv[1:5], *(int(v) for v in sys.argv[5:6]))
