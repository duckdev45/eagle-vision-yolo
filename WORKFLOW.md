# 訓練工作流設計（Training Workflow Design）

> **2026-09-19 PMS 主線補充**：日報照片工種分類與新工種候選採
> [PMS-CLASSIFICATION.md](docs/PMS-CLASSIFICATION.md) 的狀態與資料契約。
> `make pms-model`／`make pms-retrain` 與操作台共用 PMS 專用步驟，關閉 legacy 補樣；
> 人工候選、資訊不足及排除照片暫停訓練。新類核准定義後仍需逐張人工確認。

> 狀態：**v1** · 2026-08-17 · 對應 `labels.yaml` v5 · 撰寫者 duck
> 這份文件是流程的契約。程式碼與它不符時，先改文件或先改碼——但不要讓兩邊各說各話。

---

## 1. 這份文件回答三個問題

| 問題 | 現況 |
|---|---|
| 這是可重複的流程，還是某個人手動跑出來的？ | ✅ **可重複**——七個階段都是 CLI、切分凍結在 JSON、13 份實驗結果存檔在 `reports/` |
| 誰按按鈕、多久跑一次、什麼時候自動重訓？ | ⚠ **只有資料同步自動化**（cron 每日 08:00）。訓練與評估全手動、無觸發條件 |
| 出問題怎麼發現、怎麼回退？ | ❌ **未定義**。沒有 model registry、沒有上線門檻、沒有 rollback 程序 |

第 6 節是缺口清單與補齊順序。

---

## 2. 設計原則（四條鐵律）

這四條不是風格偏好，是踩過坑之後定下來的。每一條都有對應的自動化測試在 `tests/test_core.py` 擋著。

1. **`raw/` 不可變，`derived/` 隨時可重建。**
   抄下來就不改。分類規則還沒定案，下載時就分資料夾等於每改一次規則要重抓幾百張。
   結構用 symlink 放在 `derived/tree/`，重建成本為零。

2. **前處理必須與標籤完全無關。**
   所有照片一視同仁遮蔽，包含本來就乾淨的圖。否則「四角是不是灰的」本身就成了洩漏訊號。

3. **畫面幾乎相同的照片不可跨 train/test。**
   日報按 `constrId × reportDate` 整天切；QMS 按 `constructionInsId` 整格切
   （一格是同位置同時間拍的 2~3 張）。隨機切會讓測試集裡出現訓練集的近乎同一張。

4. **每個實驗都要有對照組；下判斷看分組交叉驗證，不看單一數字。**
   測試集只有 99 張，1 張 = 1.0pt。單一固定切分的數字只用於跨次比較，不用於下結論。

---

## 3. 流程：七個階段

```
S0 Ingest ─→ S1 Label ─→ S2 Prepare ─→ S3 Split ─→ S4 Train ─→ S5 Evaluate ─→ S6 Package
   每日 cron     隨時可重算    一次性        凍結        兩條路線       產出 reports/     ONNX
                                                                          │
                                                          S7 Console ◄────┘
                                                          （app.py 貫穿全程）
```

| 階段 | 指令 | 輸入 | 產物 | 契約 | 自動化守門 |
|---|---|---|---|---|---|
| **S0 Ingest** | `sync.py`（cron 08:00）<br>`qms.py --cells / --sample`<br>`legacy.py --root …`（一次性） | PMS REST API<br>QMS REST API | `data/raw/{photos,reports,manifest.csv}`<br>`data/qms/raw/{photos,cells,manifest}`<br>`data/legacy/raw/{photos,manifest}` | 只 append，用 `report_index.csv` 的 version 當水位線做增量。舊 pptx 那批**三層去重**：sha1 擋位元組相同（4,472 張）、dhash 擋重壓縮過的同一張（2,173 張）、dhash 比對 `raw/photos` 擋與 PMS 撞畫面（4 張）。pptx 一定重編碼，只靠 sha1 會漏掉一半 | `test_partial_sync_does_not_deactivate_everything`<br>`test_index_upsert_keeps_other_reports_watermark`<br>`test_anno_keys_catches_new_frontend_fields` |
| **S1 Label** | （`labels.py` 被下游引用） | `manifest.csv`<br>`labels.yaml` v5<br>`data/review.csv`（人工裁決）<br>`manifest.chipsOn` / `specKey`（人寫的參考答案）<br>`reference/qms_to_report.yaml` | 記憶體中的 `cls` 欄 | 標籤屬 derived、隨時可重算；規則**順序即優先權**；不足 `min_class_size: 12` 的類別整批 drop；`drop_fallback: true` 讓「其他」不管張數一律排除（舊 pptx 進來後它有 1,140 張、455 種互不相干的標題）。`review.csv` 是**唯一照片層級**的真標籤，蓋過 title 推出來的（連被 junk 排掉的也能救回來），append-only 留痕，**不進 derived**——重跑 pipeline 不會洗掉 | `test_labels`<br>`test_rule_order_is_the_contract`<br>`test_class_names_follow_convention`<br>`test_exclude_{bad_reports,future_dates,status}` |
| **S2 Prepare** | `prepare.py --src both` | `raw/photos` | `derived/images/*.jpg`（長邊 512） | 日報遮四角（30%×12%），但 **`reportDate >= 2026-08-14` 起上傳端存乾淨 raw，不遮**（`CLEAN_FROM`；上界取該列 `syncedAt`，擋髒的未來日期）；QMS 遮左下浮水印（0~56% × 60~100%）。兩邊遮**不同**的東西，因為兩個系統烤在圖上的東西不同。灰角＝舊資料，靠 S3 照 constrId+日期切來擋背日期 | `test_mask_corners_hits_four_corners`<br>`test_mask_watermark_covers_qms_box_only`<br>`test_clean_ids_only_takes_sane_dates_from_cutoff`<br>`test_process_mask_false_keeps_corners` |
| **S3 Split** | `split.py` / `--qms` / `--merged` / `--with-legacy` | 標籤後的 manifest | `derived/splits/{v1,qms-v1,mix-v1}.json` | `labels` 與 `classes` **凍進 JSON**——類別集合是 `min_class_size` 現算的，資料一長就變，不凍住則前後兩次評估的分母不同卻長得一樣。舊 pptx **只進 train**（測試集要代表產品實際收到的照片），且撞到 PMS 測試日的整批丟掉；`--with-legacy` 預設關，實測不提升現有類別分數（0.823 → 0.782），價值在解鎖 9 個新類別 | `test_split_by_site_and_date`<br>`test_dhash_survives_recompression_but_separates_photos`<br>`test_legacy_site_alias_matches_pms_names` |
| **S4 Train** | **路線 A**：`features.py` → `train.py`<br>**路線 B**：`finetune.py --stage1 / --stage2` | split JSON + 影像 | `models/probe-*.pkl`<br>`models/{backbone-qms,ft-report-*}.pt` | A = 凍結 SigLIP + LogisticRegression，**快、當診斷用**（分不開通常是標籤有矛盾，不是模型不夠大）。`C` 預設 300，用 GroupKFold（同工地同日不跨 fold）在 train 上選的；C=1 對 768 維 embedding 過度正則化，實測差 11pp<br>B = convnext_tiny 兩階段微調，必附 `--scratch` 對照組 | — |
| **S5 Evaluate** | `evaluate.py --split … --run …` | 模型 + split | `reports/{date}-{tag}/`<br>`metrics.json` `config.json` `confusion.png` `gemini_detail.csv` `errors/` | 測試集**全程固定**日報那 99 張；Gemini 基準線走**同一份** `labels.yaml` 正規化；`config.json` 記 `labelsVersion / split / encoder / model`；`finetune.py` 的 `bestTop1` 是拿測試集挑 epoch（模型選擇洩漏），**比較一律看 `finalTop1`** | — |
| **S6 Package** | `finetune.py`（匯出）<br>`predict.py --ckpt … --src …` | `.pt` | `.onnx` + `{ckpt}-classes.json`<br>`{ckpt}-preds-{src}.csv` | 推論前處理必須與訓練的 `eval_tf` **逐步一致**（resize 256 → center crop 224 → normalize）。ONNX 單張 CPU 19ms | — |
| **S7 Console** | `streamlit run src/app.py` | 全部 | — | 同步、看分佈、翻照片、改 `labels.yaml`、看混淆矩陣的單一入口。照片頁**最新加入的排最前面**（manifest 是 append-only，原順序等於最舊在最前）。**⑤ 歷史資料**：舊 pptx 那批的去重統計、補了哪些類別、以及**它到底有沒有進線上那組訓練集**（讀 split 的 `trainLegacy`）。照片頁下方「判斷依據」用**遮擋法**解釋線上那顆探針：遮一格 → 重新編碼 → 看答案掉多少（`explain.probe_cam`）。要看哪一顆由 `derived/splits/CURRENT` 決定（`split.current()` 讀它，`make use` 或操作台的切換按鈕寫它）。<br>**① 同步**下方的「重跑模型」= README 那條鏈的按鈕版；跑完先顯示分數，**切換是另一顆按鈕**——換掉大家看到的答案是個決定，不是重訓的副作用。<br>**④ 複核佇列** = 任一參考答案與 title 不一致的照片配上熱區，人裁決寫進 `data/review.csv`。三個參考答案照「誰寫的」排序：`chipsOn`（主任自打的查驗重點，543 條）、`specKey`（前端點選的工種，與規則一致率 98.6%）都是**人寫的**，排前面；`predWorkItem` 是 Gemini 答的，只能當提示。chips 落到 fallback 不算不一致（查驗重點寫的是驗收條件，本來就不含工種詞——那是沒訊號，不是有異議）。這是唯一能突破「工項標籤套到照片」天花板的路（實測 268 個工項有 2 張照片，常在拍不同階段） | `test_review_overrides_beat_the_title_rule`<br>`test_review_csv_roundtrip`<br>`test_human_refs_reads_chips_and_speckey` |
| **S8 Explain** | `explain.py --ckpt ft-report-*`（convnext，Grad-CAM）<br>`explain.py --probe`（探針，遮擋法，**批次**） | 模型 + 影像 | `reports/{date}-explain-*/`<br>`derived/features/cam-{model}-{split}-g{N}.npz` | 兩條路解釋**兩個不同的模型**，別混著看。`ft-report-*` 是 v5 改名前訓的，類別對不上現在的 split，跑起來會警告——熱區仍正確（它只解釋預測），但 top-1 與 truth 欄不可信。`--probe` 一次算完全部（593 張約 4 分鐘），只存 grid×grid 小陣列（200KB），疊圖是看的時候才畫 | `test_probe_cam_finds_the_block_that_matters` |

---

## 4. 已封板的結論（不需要重新討論）

| 結論 | 證據 |
|---|---|
| **QMS 那 36k 張對「日報工種辨識」沒有幫助。** 三條路都試過：混訓、純遷移、預訓練。 | 混訓 0.687 / 純遷移 0.283 / 兩階段微調 0.748，全部低於什麼都不做的 0.818。劑量曲線單調下降（0→100→342→700→1165 張：0.768→0.798→0.707→0.626→0.596） |
| **原因是領域偏移，不是資料品質。** 稽核照是「查驗點特寫」，日報照是「工人在現場施工」的廣角。 | QMS 自己學自己學得起來：線性探針 0.680 → 微調 **0.884**（n=2204，CI 僅 ±1.4pt）。編碼器一鬆綁就吃得下那個領域，只是吃不到日報這邊 |
| **統一遮罩的代價不值得，已回退。** | 兩邊都遮成「四角 + 左下」→ 日報基準線 0.818 → 0.768（−5pt）。改成兩階段微調後洩漏顧慮消失 |
| **不取代 Gemini，用信心門檻串接。** | 兩邊錯的照片重疊不高；本地模型只有 10 類，而 QMS 母體顯示實際被拍的工種有 47% 落在這 10 類之外，且模型目前沒有「以上皆非」 |

**副產品**：`backbone-qms`（0.884 / 26 個中類 / 已匯出 ONNX）本身就是一條獨立的產品線——QMS 稽核照自動分類。它不該被當成日報實驗的失敗品埋掉。

---

## 5. 量測精度：為什麼不能看單一數字

測試集 n=99，**1 張 = 1.0pt**。各實驗的 95% 信賴區間（Wilson）：

| 實驗 | top-1 | 95% CI | 區間寬度 |
|---|---|---|---|
| A 線性探針（現行最佳） | 0.818 | 0.731 – 0.882 | 15.1pt |
| 微調・不看 QMS | 0.788 | 0.697 – 0.857 | 15.9pt |
| 統一遮罩（已回退） | 0.768 | 0.675 – 0.840 | 16.4pt |
| 微調・先看 QMS | 0.748 | 0.654 – 0.823 | 16.9pt |
| C 硬合併 | 0.687 | 0.590 – 0.770 | 18.0pt |

**前四列的區間互相重疊——統計上分不開。** 以「工地 × 日期」為群組做 20 次分組交叉驗證：

```
top-1 = 0.775 ± 0.075   (範圍 0.660 ~ 0.976)
單一固定切分的 0.818 落在第 75 百分位 —— 偏樂觀
```

對比之下 QMS 那條線的 0.884（n=2204、CI ±1.4pt）才是實打實的。

**這一節是第 6 節 G1 的全部理由**：在測試集擴大到能分辨 3pt 差異之前，任何「模型變聰明了」的宣稱都不成立。

---

## 6. 缺口與補齊順序

| # | 缺口 | 為什麼重要 | 建議做法 | 優先 |
|---|---|---|---|---|
| **G1** | 沒有凍結的黃金測試集 | 99 張、CI ±8pt；且每改一次 `labels.yaml` 測試集本身就換一份（489→457→441 張是三份不同的考卷），分數不可比 | 300–500 張人工標、分層抽樣、**獨立於 `labels.yaml`**、兩人各標一次算一致度（順便得到人類天花板） | **P0** |
| **G2** | 沒有上線門檻（promotion gate） | 「什麼分數才能上線」沒有定義，等於每次都靠開會決定 | 見第 7 節。主指標改成 **coverage @ precision ≥ 0.90**，判定一律用分組 CV 而非單點 | **P0** |
| **G3** | 沒有線上回饋迴路 | 使用者在產品裡修正工種的行為 = 免費的真實標籤，目前沒有收集 | 產品端記 override log（改前/改後/modelVersion）→ 回流成訓練資料與線上準確率的代理指標 | **P0** |
| **G4** | 沒有自動重訓觸發 | 只有 `sync.py` 有 cron，其餘全手動；資料長了沒人知道該重訓 | 條件觸發：新增 ≥ 200 張 **或** 覆寫率週對週上升 ≥ 5pt **或** 每月一次，取先到者 | P1 |
| **G5** | 沒有 model registry / rollback | `models/` 是一坨檔名，沒有「現在線上是哪一個」的指標，也沒有版本鏈 | `models/registry.json` 記每個候選的 metrics/gate 結果/上線時間 + 一個 `current` 指標。rollback = 改一個欄位 | P1 |
| **G6** | `config.json` 沒記 code 版本 | 無法重現「那次實驗是哪份程式碼」 | 加 `gitSha` / `uv.lock` hash / `seed` / `pythonVersion` | P1 |
| **G7** | 14 個測試要手動跑 | 洩漏防護只在有人記得跑的時候有效 | pre-commit hook 或 GitHub Action 跑 `tests/test_core.py` | P2 |

---

## 7. 目標狀態：重訓迴路與上線門檻

### 7.1 迴路

```
      ┌─────────────────────── 每日 ────────────────────────┐
      │                                                     │
  sync.py (cron)                                    產品線上推論
      │                                              （影子模式）
      ▼                                                     │
  資料累積 ────► 觸發條件成立？ ──否──► 等                    │
                   │是                                       │
                   ▼                                         │
             prepare → split → train → evaluate              │
                   │                                         │
                   ▼                                         │
            promotion gate 全過？ ──否──► 留在 reports/，不上 │
                   │是                                       │
                   ▼                                         │
            寫入 registry → 灰度 → current ──────────────────┘
                                                    │
                                          override log 回流 ──► 資料累積
```

**觸發條件**（任一成立，取先到者）：新增訓練照片 ≥ 200 張 · 線上覆寫率週對週上升 ≥ 5pt · 距上次重訓滿一個月。

### 7.2 Promotion gate（草案，數字待 G1 完成後校準）

候選模型要上線，必須**全部**通過：

| 檢查 | 門檻 | 為什麼 |
|---|---|---|
| 黃金測試集 coverage @ precision ≥ 0.90 | 不低於現行版本 − 2pt | 產品指標，不是 top-1 |
| 分組交叉驗證 macro-F1 | 不低於現行版本 − 1σ | 防止單點運氣 |
| 每一類的 recall | 沒有任何類別掉到 0 | 防止小類被犧牲換總分 |
| 洩漏測試 | `tests/test_core.py` 全綠 | 切分與遮罩的鐵律 |
| 前處理一致性 | 離線 vs ONNX 同一批照片預測一致率 ≥ 99.5% | 線上/離線落差最常見的來源 |
| 影子模式對照 | 與現行版本在同一批線上照片上的分歧率 < 15%，且分歧樣本人工抽檢 30 張後判定不更差 | 上線前最後一道 |

### 7.3 上線與回退

- **上線形態**：獨立小服務（FastAPI + onnxruntime），掛在現在呼叫 Gemini 的那個點旁邊。獨立才能獨立回退。
- **推論輸出必須帶版本**：`{cls, prob, modelVersion, labelsVersion, encoderVersion}` 全部寫進 DB。沒有這個，三個月後無法回答「這張是哪個版本判的」。
- **串接而非取代**：`prob ≥ τ` 用本地模型，否則打 Gemini。τ 由影子模式資料校準。
- **回退**：把 `registry.json` 的 `current` 改回上一版；或把 τ 調到 1.0——等於完全回到現在純 Gemini 的狀態。回退是改一個數字，不是重新部署。
- **監控**：各類別的使用者覆寫率（線上真實準確率的免費代理）· 低信心比例的走勢（突然上升 = 進到新工序或新建案）· 預測分佈 vs 訓練分佈的漂移。

---

## 8. 角色與節奏（待填）

| 項目 | 目前 | 需要決定 |
|---|---|---|
| 誰跑重訓 | duck（手動） | 自動化後由誰看結果、誰簽 gate |
| 誰標黃金測試集 | 無 | 需要兩位懂現場的人各標一次（G1 的前提） |
| 誰決定上線 | 無 | promotion gate 全過是否即可上，或仍需人簽 |
| 產品端 override log | 無 | 需要 PMS 端排程配合（G3 的前提） |

---

## 附錄：一次完整重跑

```bash
uv sync
uv run src/sync.py                              # S0
uv run src/qms.py --sample 12000                # S0（QMS，選用）
uv run src/prepare.py --src both --force        # S2
uv run src/split.py                             # S3
uv run --extra train src/features.py            # S4-A
uv run --extra train src/train.py               # S4-A
uv run --extra train src/evaluate.py --run A-baseline   # S5
uv run tests/test_core.py                       # 守門
```
