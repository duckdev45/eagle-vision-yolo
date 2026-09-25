# 把 README「新照片進來之後」那條鏈包成一個指令。
#
#   make retrain SPLIT=v9     照片有新的 → 從同步一路跑到熱區
#   make model   SPLIT=v9     只改了 labels.yaml 或複核裁決 → 從切分開始
#
# SPLIT 每次要給新名字，別原地覆蓋——標籤一變，同名舊報告的分母就對不上了。

.DEFAULT_GOAL := pms-app

SPLIT ?= v8
LEGACY ?=                       # LEGACY=1 → 把舊 pptx 那批**全部**加進 train（會多 9 類）
MIN_TRAIN ?= 12                 # 稀有類別訓練保底：含它的測試日整天搬回 train
LEGACY_FILL ?= 40               # 切完還不足的類別，只從 legacy 補那幾類到這個數
TRAIN := uv run --extra train
SPLIT_FLAGS := $(if $(LEGACY),--with-legacy,) \
               $(if $(MIN_TRAIN),--min-train $(MIN_TRAIN),) \
               $(if $(LEGACY_FILL),--legacy-fill $(LEGACY_FILL),)

.PHONY: help retrain model data cams test lint fmt app sync use legacy journal queue newclass qs qs-phases fr-demo fr-gdino cvat-export g1-sample yolo-dataset contract-priority
.PHONY: pms-app pms-status pms-candidates pms-export pms-import pms-model pms-retrain pms-ai

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

pms-retrain: ## 同步 PMS 後訓練；不自動切換目前模型
	$(TRAIN) src/pms.py train --name "$(SPLIT)" --sync

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/	/'

retrain: data model  ## 新照片進來之後的完整重跑

data: ## 1~3：同步、遮蔽、抽 embedding（都是增量，只做新檔）
	uv run src/sync.py
	uv run src/prepare.py
	$(TRAIN) src/features.py

model: ## 4~7：重切、重訓、評估、重算熱區。改標籤或裁決之後跑這個就夠
	$(TRAIN) src/features.py --crops
	uv run src/split.py --name $(SPLIT) $(SPLIT_FLAGS)
	$(TRAIN) src/train.py --split $(SPLIT)
	$(TRAIN) src/evaluate.py --split $(SPLIT) --run $(SPLIT)
	$(TRAIN) src/explain.py --probe --split $(SPLIT)
	uv run src/journal.py
	@echo
	@echo "⚠ 還沒切換。分數看過覺得可以，再跑：make use SPLIT=$(SPLIT)"
	@echo "  故意分兩步——換掉大家看到的答案是個決定，不是 make 的副作用。"
	@echo "  這版學到什麼、哪些混淆是慣犯：reports/JOURNAL.md"

use: ## 把操作台切到 SPLIT 指的那一版（跑完 model、看過分數再用）
	uv run python -c "import sys;sys.path.insert(0,'src');import split;split.set_current('$(SPLIT)');print('操作台已切到', split.current())"

journal: ## 只重寫 reports/JOURNAL.md（不重算任何東西，隨時可跑）
	uv run src/journal.py

queue: ## 印出還沒裁的複核佇列（裁決要人做，在 make app 的 ④）
	$(TRAIN) src/review.py

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

LEGACY_ROOT ?= ~/Downloads/115年日報表

legacy: ## 一次性：從舊 pptx 抽照片（跨來源去重）→ 前處理 → 抽 embedding
	$(TRAIN) src/legacy.py --root "$(LEGACY_ROOT)"
	uv run src/prepare.py --src legacy
	$(TRAIN) src/features.py --src legacy
	@echo
	@echo "抽完了。要把它加進訓練：make model SPLIT=vN LEGACY=1"
	@echo "  預設不加——實測它不提升現有類別的分數，價值在解鎖 9 個新類別。"

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

fr-gdino: ## 樂氧森缺失照片 × GDINO 煙霧測試（--all 全量；預設抽 30）
	uv run --extra gdino src/fr_gdino.py --all

fr-demo: ## GDINO 標註 demo 操作台（照片牆＋人審打分）
	uv run streamlit run src/fr_app.py

cvat-export: ## 缺失框冷啟動：照片按案場×日期打成 CVAT task manifest（--source pms|field|golden）
	uv run src/cvat_export.py --source pms --sample 300

g1-sample: ## G1 黃金集抽樣：trade 400 分層＋defect 樣態保底 30（不足標本輪不評）
	uv run src/g1_sample.py

yolo-dataset: ## defects.csv 的 CVAT 框 → YOLO 資料集＋imgsz 量測（--measure-gdino 只量測）
	uv run src/yolo_dataset.py --out v1-defects

contract-priority: ## 合約收集優先序重排（97 項合約相依 REQUIRED）
	uv run src/contract_priority.py

jev-eval: ## Jev vs labels.yaml regex A/B（真標籤 = review.csv 人工裁決）
	uv run src/jev_label_eval.py $(ARGS)

jev-fallback: ## regex 沒有意見那桶（drop_fallback），Jev 救得回多少
	uv run src/jev_label_eval.py --fallback-probe $(ARGS)

jev-doc: ## Jev 探測結果 → 看圖複核活文件（規則補丁提案＋真標籤矛盾）
	uv run src/jev_review_doc.py $(ARGS)

jev-orphan: ## 孤兒照片（QS 有標準、labels.yaml 沒類）→ 看圖複核活文件
	uv run src/jev_orphan_doc.py $(ARGS)

jev-policy: ## Jev 判定「哪些工種該現在寫規則」（政策合成在程式碼，模型只答語義）
	uv run src/jev_newclass_policy.py

jev-newclass-doc: ## 把 Jev 說它決定不了的那 148 張做成看圖複核文件
	uv run src/jev_newclass_doc.py
	@echo "  open reports/2026-09-19-jev/NEWCLASS-REVIEW.html"
