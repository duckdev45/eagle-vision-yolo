"""公司 ISO（QS）品質標準 CLI。真實實作已上移至 core/qs_data.py（服務層）。

保留本檔的介面不變：make qs / make qs-phases / make test 的入口都是這裡。
"""

from __future__ import annotations

import os
import sys

from core.qs_data import *
from core.qs_data import main

if __name__ == "__main__":
    raise SystemExit(main())
