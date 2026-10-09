"""合約工作約定 CLI。真實實作已上移至 core/contractdata.py（服務層）。

保留本檔的介面不變：make contract / make test 的入口都是這裡。
"""

from __future__ import annotations

import os
import sys

from core.contractdata import *
from core.contractdata import main

if __name__ == "__main__":
    raise SystemExit(main())
