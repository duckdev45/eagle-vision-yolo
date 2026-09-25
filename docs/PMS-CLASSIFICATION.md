# PMS 日報照片工種分類與新工種候選

本版範圍：2026-09-19 起，以 PMS 日報照片為主來源，完成看圖複核、工種分類、未知照片整理、新類核准與分類器重訓。缺失照片仍在累積，缺失偵測與 YOLO 不屬於這次流程。

## 入口與資料流

`make pms-app` 啟動 `src/app.py`。PMS 頁籤為「②資料總覽」「③照片工種」「⑤新工種候選」，原本的證據框操作保留在「④進階複核」。

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

候選詞群組重用 `newclass.candidates`，但只統計有效 PMS 未知照片，並顯示案場數與日期數。這是用字群組提示；是否具備一致的圖像特徵仍需看圖判斷。同標題不等於同工種。

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
| `src/pms.py` | 盤點、審閱包、單張裁決與訓練命令 |

`data/review.csv` 繼續保存人工標籤及原有證據框。新增的 `data/pms_review.sqlite3` 保存封包、AI 建議、人工狀態、候選、核准定義與事件歷史。兩者都不是 derived 檔案，備份／搬機需一起保存。

目前採單機操作台模式。SQLite 的事件批次使用交易，但 CSV 與 SQLite 不是跨檔案的單一交易，不把此版當多人同時標註服務。舊介面重新裁決會使該照片先前的暫緩狀態失效，讓既有流程可以恢復照片。

分類表版本是內容雜湊，新 split 保存當時完整分類表與版本。新類先透過人工照片標籤累積樣本，無須臆造 regex；需要自動路由新寫法時，再獨立修改 `labels.yaml` 並檢查規則順序。

## PMS 訓練與驗證

```bash
# SPLIT 請取尚未使用的新名稱
make pms-model SPLIT=pms-v40
make pms-retrain SPLIT=pms-v41
# 檢查新報告後才執行
make use SPLIT=pms-v40
```

兩個 PMS 命令與操作台共用同一串步驟，差異只有是否先同步遠端。本機前處理與特徵先增量補齊，再切分、訓練、評估、產生解釋與學習紀錄。明確指定 `with_legacy=False, legacy_fill=0`，訓練母體只收 `WORK_ITEM`，依案場與日期整組切；出工照即使有舊人工裁決或證據裁切也不進新切分。候選／暫緩／排除照片不入訓。特徵不足、空測試集或類別不足時停止，不自動切換模型。失敗時可能已產生該次 split，修正後請用新版本名。

既有通用 `make model` 與 legacy/QMS 實驗路徑保留，本版操作使用 `pms-*` 命令。現行模型若原本含 legacy 訓練資料，需以 PMS 專用流程重訓並明確切換後，才會變成純 PMS 模型。

測試使用 pytest 暫存資料，驗證來源邊界、答案分離、審閱包、人工改判／暫緩／恢復、候選生命週期、PMS 訓練路由與 Streamlit 互動。程式測試不代表分類模型變準，準確率仍需獨立人審測試集。

`calibration` 直接用純標題規則對照人工答案，避免先覆寫答案後比較的假 100%。這個歷史已裁子集有選樣偏差，不能當自動確認門檻。
