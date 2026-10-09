# 日常不用手動：`make daily`（hermes cron 每天跑）＝同步→分流→夠多新資料就重訓→公平考卷過關自動切換。
# 手動重訓走同一條鏈（pipeline/pms_workflow.py）：
#
#   make retrain SPLIT=v46    照片有新的 → 同步後訓練（＝pms-retrain）
#   make model   SPLIT=v46    只改了 labels.yaml 或裁決 → 用本機照片訓練（＝pms-model）
#
# SPLIT 每次要給新名字，別原地覆蓋——標籤一變，同名舊報告的分母就對不上了。

.DEFAULT_GOAL := pms-app

# 匯入路徑只靠 .venv 的 editable 安裝（pyproject [tool.hatch.build]）。外部帶進來的 PYTHONPATH
# 會把別的 Python 版本的 site-packages 排在 .venv 前面——實測 Hermes 起的子行程帶著 3.14 的
# site-packages，numpy 直接載不起來。每日排程無人值守，在這裡一律清掉。
unexport PYTHONPATH PYTHONHOME

SPLIT ?= v8
TRAIN := uv run --extra train

.PHONY: help retrain model cams test lint fmt app sync use journal queue newclass qs qs-phases cvat-export g1-sample yolo-dataset defect-probe contract-priority label-pack
.PHONY: service-image daily route pms-app pms-status pms-candidates pms-export pms-import pms-model pms-retrain pms-ai pms-quality

daily: ## 每日編排：同步→特徵→分流→夠多新資料就重訓→公平考卷過關自動切換（排程跑這個）
	$(TRAIN) src/daily.py $(ARGS)

route: ## 只重算分流（收件匣）；不同步、不訓練
	uv run src/daily.py --only-route

pms-quality: ## 盤點 PMS 原圖、前處理圖與訓練影響（唯讀原圖）
	uv run src/pms_quality.py

pms-ai: ## OpenAI 看圖與標題分類；ARGS='--limit 4'，只保存建議
	uv run src/pms.py ai $(ARGS)

pms-app: ## PMS 工種操作台（基本相依即可看圖與人工分類）
	uv run streamlit run src/app.py

pms-status: ## PMS 照片分類與待複核盤點（唯讀）
	uv run src/pms.py status

pms-candidates: ## PMS 新工種候選與未知標題群組（唯讀）
	uv run src/pms.py candidates

pms-export: ## 匯出看圖審閱包；ARGS='--out data/pms-review/batch.zip --limit 12'
	uv run src/pms.py export $(ARGS)

pms-import: ## 匯入 AI 建議；ARGS='--input data/pms-review/response.json'
	uv run src/pms.py import $(ARGS)

pms-model: ## 只用本機 PMS 照片訓練；SPLIT 必須是新版本名
	$(TRAIN) src/pms.py train --name "$(SPLIT)"

pms-retrain: ## 同步 PMS 後訓練；不自動切換（自動切換只走 make daily 的公平考卷）
	$(TRAIN) src/pms.py train --name "$(SPLIT)" --sync

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/	/'

retrain: pms-retrain ## 新照片進來之後的完整重跑（同 pms-retrain）

model: pms-model ## 改標籤或裁決之後重訓（同 pms-model）

use: ## 手動把操作台切到 SPLIT 那一版（自動切換走 make daily 的公平考卷）
	uv run python -c "from core import model_registry as r;r.set_current('$(SPLIT)');print('操作台已切到', r.current())"

journal: ## 只重寫 reports/JOURNAL.md（不重算任何東西，隨時可跑）
	uv run src/journal.py

queue: ## 印出待人看的照片（與收件匣同一份分流結果；ARGS='--reason 標題沒有對應規則'）
	uv run src/review.py $(ARGS)

newclass: ## 沒被規則認領的照片 → 下一批新工種的候選詞
	uv run src/newclass.py

qs: ## 公司 QS 標準統計（A~E 佔比、請款靶、合約相依、各份工序階段）
	uv run src/qsdata.py

qs-excel: ## QS 標準 + 檢查項匯出 Excel（reports/QS標準總表.xlsx，給同事查閱）
	uv run src/qs_excel.py

contract: ## 合約工作約定統計（付款節點、罰則、驗收數值、QS 交叉引用）
	uv run src/contractdata.py

qs-phases: ## 重新產出 reference/iso/phases.yaml（改了 raw/*.tsv 之後跑）
	uv run src/qsdata.py --emit-phases

cams: ## 只重算熱區快取（換了模型才需要）
	$(TRAIN) src/explain.py --probe --split $(SPLIT)

sync: ## 只抓新日報與照片
	uv run src/sync.py

test: ## 自檢（含 QS 資料層 + 合約資料層；reference/ 為公司資料不入 git，缺席則跳過）
	uv run --with pytest pytest tests/ -q -rs
	@if [ -d reference/iso/raw ]; then uv run src/qsdata.py --self-check; \
	else echo "⚠ reference/ 不在（公司資料不入 git）——跳過 QS 自檢"; fi
	@if [ -d reference/contract/raw ]; then uv run src/contractdata.py --self-check; \
	else echo "⚠ reference/ 不在（公司資料不入 git）——跳過合約自檢"; fi

lint: ## Ruff lint + format 檢查（CI 用：只查不改）
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Ruff 一鍵修正（lint --fix + format；改動自行 review）
	uv run ruff check . --fix
	uv run ruff format .

app: ## 操作台
	$(TRAIN) streamlit run src/app.py

cvat-export: ## 缺失框冷啟動：照片按案場×日期打成 CVAT task manifest（--source pms|field|golden）
	uv run src/cvat_export.py --source pms --sample 300

g1-sample: ## G1 黃金集抽樣：trade 400 分層＋defect 樣態保底 30（不足標本輪不評）
	uv run src/g1_sample.py

yolo-dataset: ## defects.csv 的 CVAT 框 → YOLO 資料集＋imgsz 量測（--measure-gdino 只量測）
	uv run src/yolo_dataset.py --out v1-defects

defect-probe: ## 缺失改善弱標籤 embedding baseline → CVAT 標註優先序（reports/defect-probe/）
	$(TRAIN) src/defect_probe.py $(ARGS)

contract-priority: ## 合約收集優先序重排（97 項合約相依 REQUIRED）
	uv run src/contract_priority.py

service-image: ## 建推論服務 image；預設用 models/service/CURRENT（考卷過關自動更新），BUNDLE=v43 可指定
	@b="$(or $(BUNDLE),$$(cat models/service/CURRENT 2>/dev/null))"; \
	if [ -z "$${b}" ] || [ ! -f "models/service/$${b}/metadata.json" ]; then \
	  echo "❌ 找不到服務包 models/service/$${b}（先 make daily 過關匯出，或 BUNDLE=<版本>）"; exit 1; fi; \
	echo "→ eagle-vision-api:$${b}"; \
	docker build -f service/Dockerfile --build-context bundle="models/service/$${b}" -t "eagle-vision-api:$${b}" .

label-pack: ## 匯出去識別化判準包（白名單＝docs/DATA-BOUNDARY.md）；VERSION=p1，授權已簽再加 LICENSED=1
	uv run src/export_label_pack.py --version "$(VERSION)" $(if $(LICENSED),--licensed,)
