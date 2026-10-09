# PMS 日報照片工種分類與新工種候選

本版範圍：2026-09-19 起，以 PMS 日報照片為主來源，完成看圖複核、工種分類、未知照片整理、新類核准與分類器重訓。缺失照片仍在累積，缺失偵測與 YOLO 不屬於這次流程。

## 入口與資料流

`make pms-app` 啟動 `src/app.py`。2026-10-09 起主導覽只有四格：**收件匣**（每日分流後要人看的照片，一次一張一鍵裁決）、
**總覽**（排程狀態、自動切換紀錄、類別盤點）、**報告**、**進階**。逐張工作台（照片工種）、證據框（進階複核）、
照片工種、進階複核（含標框）、新工種候選、標籤規則、規範庫都收在「進階」；三處「要不要人看」都讀同一份每日分流。確認者預設讀 `.env` 的 `PMS_REVIEWER`。

```text
PMS 同步 → manifest + 原照片 → 工種工作台
                                ↓
              標題規則 / 圖像預測 / 人工答案
                                ↓
                 看圖確認，可搭配 AI 審閱包
                     ┌──────────┴────────┐
                  既有類別             新工種候選
                     │         定義 / 排除條件 / 分類依據
                     │                   ↓ 核准
                     │             完整分類表增加新類
                     └──────────┬────────┘
                            逐張人工確認
                                ↓
                          data/review.csv
                                ↓
                   PMS 分組切分 → 訓練 → 評估
                                ↓
                            明確切換版本
```

`core/pms_review.py` 組裝有效 PMS **施作項目**母體：經 `core/pms_source.work_items()` 只保留 `source=WORK_ITEM`，再排除 inactive、非 PMS dataset 與既有壞日報條件。`WORKFORCE` 是出工紀錄，公工／打石工等 `tradeName` 不是施作項目標籤；此處依來源欄位篩選，不用職稱或標題關鍵字排除。來源不明的列不進分類；整份 manifest 缺 `source` 時要求重新同步，不猜測。

工作台、複核佇列、PMS 候選挖掘與訓練母體共用這個邊界；原始出工照片、manifest 與既有裁決保留。只有 `WORK_ITEM` 範圍內的空標題、junk 標題、fallback、小樣本類別留在工作台供看圖。是否能訓練由標籤與人工狀態另外決定。

## 答案來源與人工狀態

| 欄位／紀錄 | 意義 | 是否照片真值 |
|---|---|---|
| `ruleClass` | 純標題規則結果，未套人工覆寫 | 弱標籤 |
| `modelClass` | 目前圖像分類器的預測 | 否 |
| `suggestion` | 看圖建議、圖片證據、理由與來源 | 否 |
| `humanClass` | `review.csv` 的最後一次有效人工裁決 | 是 |

工作台顯示真正的 train／test／unseen，不把「不在 test」的照片全部稱作訓練照。完整類別表包含 `labels.yaml` 所有類別與已核准的新類；模型尚未學會的類別，仍能被人工選取。

照片狀態為 `pending`、`classified`、`candidate`、`uncertain`、`excluded`。候選、資訊不足或非本輪語料會暫停進入訓練；原圖與舊裁決保留，重新確認類別後可恢復。

樣本門檻取 `Labeler.min_class_size`。達門檻表示可參與下輪訓練，不代表準確度合格、已經重訓或已切換模型。有效標籤統計包含人工答案與尚未複核的標題規則結果。

## 新工種候選

候選先記錄代表照片、共同可見特徵與名稱。核准時需要完整定義、排除條件與分類樹／QS 依據。這些文字由確認者提供，系統不自動推導或宣稱已驗證 QS 內容。

核准定義只增加完整分類表，不會批次替代表照片貼標；每張仍需回照片頁確認。既有類別不能被重複提出成新類。同一候選可在待核准期間追加照片，不開新類時仍保留結案紀錄。

候選詞群組重用 `core/rule_candidates.candidates`（`make newclass` 同一份），但只統計有效 PMS 未知照片，並顯示案場數與日期數。這是用字群組提示；是否具備一致的圖像特徵仍需看圖判斷。同標題不等於同工種。

## AI 自動分類：圖片與日報標題

「③ 照片工種」與「④ 進階複核」均有 **AI 自動分類：照片＋日報標題** 面板。選圖後按「AI 看圖＋標題分類」，才透過 OpenAI Responses API 送出該批照片與文字；每批預設選四張、最多十二張。單純開頁或重新整理不會呼叫 API。

在專案 `.env` 設定下列值：

```dotenv
OPENAI_API_KEY=your-api-key
PMS_OPENAI_MODEL=your-image-and-structured-output-model-id
```

模型 ID 需是帳號實際可用、支援圖片輸入與結構化輸出的模型，也可在畫面覆寫。金鑰不在 UI 顯示。API 用量與 ChatGPT 訂閱分開計費；沒有 API 設定時仍可使用下方本機審閱包。

每張回覆分開保存 `visualEvidence`（圖片證據）、`titleEvidence`（標題訊息）、`titleRelation`（一致／衝突／資訊不足），並依完整分類表提出既有類別或新類候選。照片可能拍到標題所述工項的前置或收尾，AI 不必強行照抄標題，也不把工人職稱當圖片答案。

自動化範圍是**產生分類建議**。AI 不改 `review.csv`，也不直接建立核准的新類。填確認者並按「確認採用此 AI 類別」後，才保存照片真值；新類候選仍需核准定義及逐張確認。

請求重用審閱包的圖片、上下文與分類表雜湊。結果保留模型與回覆 ID、提示版本；同輸入可沿用快取，改圖片／標題／分類表／模型或勾重新辨識時才重新請求。單張失敗會列出原因，已成功的結果保留；再次執行可只補失敗的照片。未完成、拒答、非法類名或來源已變更的回覆不採用。

```bash
make pms-ai ARGS='--limit 4'
# 精確指定照片或模型；--force 可重新辨識已快取結果
uv run src/pms.py ai --ids id1,id2 --model your-model-id
```

API 實作依據：[圖片輸入](https://developers.openai.com/api/docs/guides/images-vision)、[結構化輸出](https://developers.openai.com/api/docs/guides/structured-outputs)。串接以模擬 HTTP 回覆測試請求與資料寫入契約；模型在本案照片上的準確度仍需用實際辨識與人工覆核量測。

## 本機看圖審閱包

本機檔案交換流程保留。使用者選圖建立 ZIP，將圖片及上下文交給支援看圖的審閱工具，再把結果 JSON 匯回。此匯出與匯入流程本身不呼叫遠端 API。

封包包含 `images/*.jpg`、`catalog.json`、`context.json`、`PROMPT.md`、`packet.json`、`response-template.json`。圖片校正方向、保留至長邊 1600px、實際編碼為 JPEG，不改原圖。僅讀文字、未讀圖片的回覆不能當看圖結果。

每筆建議只能是既有類別、新類候選、資訊不足或非本輪語料之一，並須附可見證據與理由。新類候選須附定義，不用模型自己填的信心代替實測正確率。

匯入前驗證封包識別碼、完整照片集合、圖片內容、日報上下文與分類表版本。漏圖、多圖、重複照片、過期內容或非法類名整批拒收。匯入只新增建議，重複匯入相同建議不會增加紀錄。採用建議前再次驗證圖片、上下文與分類表。

```bash
make pms-status
make pms-candidates
make pms-export ARGS='--out data/pms-review/batch-01.zip --limit 12'
make pms-import ARGS='--input data/pms-review/response-01.json'
uv run src/pms.py calibration
```

也可用 `export --ids id1,id2 --out ...` 精確選圖，每批最多 50 張。

## 儲存與模組

| 模組 | 責任 |
|---|---|
| `core/pms_store.py` | SQLite 追加事件、最後狀態及候選版本檢查 |
| `core/pms_source.py` | PMS 施作項目 WORK_ITEM 共用來源篩選 |
| `core/pms_review.py` | PMS 照片快照、人工裁決、分類表與候選 |
| `core/pms_exchange.py` | 審閱包與建議回覆驗證 |
| `core/pms_vision.py` | OpenAI 圖片與標題請求、分類結果、快取與失敗處理 |
| `src/ui/pms_ai.py` | 照片頁與複核佇列共用的 AI 分類／人工採用操作 |
| `src/ui/pms_workbench.py` | 照片分類、候選及審閱操作 |
| `pipeline/pms_workflow.py` | CLI 與操作台共用的 PMS 訓練步驟 |
| `core/routing.py` | 每日分流（自動確認／抽查／人工佇列／隔離）、收件匣、寫回入口、待審清單匯出 |
| `core/promotion.py` | 公平考卷與自動切換門檻 |
| `src/daily.py` | 每日無人值守編排（`make daily`） |
| `src/ui/inbox.py` | 收件匣與總覽頁的排程狀態 |
| `src/pms.py` | 盤點、審閱包、單張裁決與訓練命令 |

`data/review.csv` 繼續保存人工標籤及原有證據框。新增的 `data/pms_review.sqlite3` 保存封包、AI 建議、人工狀態、候選、核准定義與事件歷史。兩者都不是 derived 檔案，備份／搬機需一起保存。

目前採單機操作台模式。SQLite 的事件批次使用交易，但 CSV 與 SQLite 不是跨檔案的單一交易，不把此版當多人同時標註服務。舊介面重新裁決會使該照片先前的暫緩狀態失效，讓既有流程可以恢復照片。

分類表版本是內容雜湊，新 split 保存當時完整分類表與版本。新類先透過人工照片標籤累積樣本，無須臆造 regex；需要自動路由新寫法時，再獨立修改 `labels.yaml` 並檢查規則順序。

## PMS 訓練與驗證

### 照片品質與人工考卷

`make pms-quality` 逐張檢查 manifest 對應的原圖與前處理圖，產出
`reports/pms-photo-quality.csv` 和 `reports/pms-photo-quality-summary.json`；若目前訓練母體有壞圖則回傳非零狀態。
同步會重新下載缺檔、過小或無法解碼的照片，只有新下載內容確認可解碼後才替換本機壞檔，舊位元組留在
`data/raw/quarantine/`。未出現在目前 API 日報清單的舊 dev 資料只列在品質報告，不會憑空補抓。

G2 上線門檻只接受**至少 300 張、由人仲裁的 PMS WORK_ITEM 照片**，而且與候選版及現行版的
訓練／測試資料在「案場 × 日期」層級完全分離。缺答案、少張數、混入 legacy、缺 embedding 或同日洩漏都會拒絕評估。
現有 `data/golden/g1_manifest.csv` 的工種抽樣以 legacy 為主，不能直接拿來宣稱 PMS 模型通過 G2；
需要另收獨立 PMS 考卷並完成人工雙標與仲裁。G2 在同一批獨立黃金折上比較兩版已訓練模型，
不在考卷上重訓。G2 未完成前不應以單次 split 分數作上線依據——每日排程的自動切換用的是下一節的公平考卷，不是 split 分數。

### 每日分流與自動切換（2026-10-09）

`make daily`（`src/daily.py`）每天在本機跑一次：同步 → 前處理／特徵 → 分流 → 夠多新資料才重訓 → 考卷過關才切換。
任一步失敗不中斷後面能做的事，結果附加到 `data/daily-log.jsonl`。

**分流**（`core/routing.py`）三個訊號：純標題規則、圖像模型（工項融合＋打底／粉光分層，與工作台同一套）、人工。

| 桶 | 條件 | 誰處理 |
|---|---|---|
| 自動確認 | 規則＝模型、信心 ≥ 0.9、沒有依定義要看圖的旗標 | 無人；**不寫 `review.csv`** |
| 抽查 | 自動確認裡固定 5%（依 fileId 雜湊，天天同一批） | 人；量自動桶的真實準確率 |
| 人工佇列 | 規則≠模型、標題無規則、模型沒學過此類、階段待人工、缺失改善照、信心不足 | 人（收件匣） |
| 隔離 | 沒原圖、沒特徵、日報日期不合理 | 修資料，不是人裁 |

兩條不可退讓的規則（`tests/test_routing.py` 擋著）：

1. **模型訊號不能來自背過那張照片的模型。** split 裡沒人審的照片本來就用規則答案訓練，拿同一顆模型看它，「規則＝模型」是背出來的。
   split 內的照片一律用 5 折分組 out-of-fold（案場×日期整組），只有 split 外的新照片用上線模型。
2. **自動確認不是人審。** 不寫 `review.csv`、不進判準包 `humanClass`、不當考卷。分流只決定誰要看，不決定答案。

**自動切換**（`core/promotion.py`）：公平考卷＝候選版 test − 現行版 train，限現行版認得的類；兩版用同一套預測各考一次。
非劣性門檻：考卷 ≥ 100 張；top1 與 macroF1 不低於現行版 − 1pt；考卷上 ≥ 10 張的類別召回不得掉超過 15pt；分類表版本須一致。
過關就寫 `CURRENT` 並匯出服務包（`models/service/<版本>`），每次都記在 `data/promotion-log.jsonl`。
重訓觸發：現行模型之後新增 ≥ 20 筆人審，或距上次訓練 ≥ 7 天且有 ≥ 50 張新照片；無人值守重訓跳過圖像解釋（操作台會現算熱區）。

收件匣的待審清單同時匯出成 `data/exports/queue/queue.json`，給未來的標註平台用，契約見 `docs/QUEUE-CONTRACT.md`。

```bash
# SPLIT 請取尚未使用的新名稱
make pms-model SPLIT=pms-v40
make pms-retrain SPLIT=pms-v41
# 檢查新報告後才執行
make use SPLIT=pms-v40
# 或交給每日排程：自動取版本名、考卷過關才切換
make daily
```

兩個 PMS 命令與操作台共用同一串步驟，差異只有是否先同步遠端。本機前處理與特徵先增量補齊，再切分、訓練、評估、產生解釋與學習紀錄。明確指定 `with_legacy=False, legacy_fill=0`，訓練母體只收 `WORK_ITEM`，依案場與日期整組切；出工照即使有舊人工裁決或證據裁切也不進新切分。候選／暫緩／排除照片不入訓。特徵不足、空測試集或類別不足時停止，不自動切換模型。失敗時可能已產生該次 split，修正後請用新版本名。

`make model`／`make retrain` 已改為 `pms-model`／`pms-retrain` 的別名（2026-10-09），legacy／QMS 實驗路徑移除，資料封存見 `data/archive/README.md`。

測試使用 pytest 暫存資料，驗證來源邊界、答案分離、審閱包、人工改判／暫緩／恢復、候選生命週期、PMS 訓練路由與 Streamlit 互動。程式測試不代表分類模型變準，準確率仍需獨立人審測試集。

`calibration` 直接用純標題規則對照人工答案，避免先覆寫答案後比較的假 100%。這個歷史已裁子集有選樣偏差，不能當自動確認門檻。
