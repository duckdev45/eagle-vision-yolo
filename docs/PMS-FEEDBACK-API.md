# PMS 現場回饋與模型回訓 API 草案

## 目標

工地與品管使用者在既有 PMS／EagleField 畫面查看照片、接受或改判工種；
Streamlit 保留給資料管理與進階複核。現場操作寫入結構化事件，經品管確認後才成為訓練真值。
此設計借鑑 AfterQuery「專家判斷 → 可驗證資料 → 評估 → 訓練 → 部署」的閉環，
不需要每位使用者各自訓練一個模型。

## 使用者角色與權限

| 角色 | 現場能做什麼 | 對訓練的效力 |
|---|---|---|
| 現場人員 | 接受預測、改判既有工種、提出新工種候選、標記照片無法使用 | 回饋事件；尚非真值 |
| 品管專家 | 查看原圖、日報與回饋，確認或退回 | 確認後可進下次訓練快照 |
| 仲裁者 | 處理不同專家意見、核准新類定義 | 決定爭議真值與分類表版本 |
| 模型管理者 | 啟動重訓、檢查固定考卷與上線門檻、切換或回退 | 管理模型版本；不改寫人工答案 |

使用 PMS 既有登入與權限；API 只相信伺服器驗證出的 actorId／角色，不相信請求內自報身分。

## 最小資料契約

`POST /v1/pms/photo-feedback`，要求 `Idempotency-Key`。請求包含：

```json
{
  "fileId": "PMS-fileId",
  "modelVersion": "v40",
  "catalogVersion": "分類表雜湊",
  "action": "correct",
  "selectedClass": "泥作-打底",
  "reasonCode": "wrong_trade",
  "evidenceBoxes": [[100, 200, 650, 800]],
  "note": "畫面主體為牆面打底"
}
```

現場畫面顯示四個主要操作：「正確」「改成既有工種」「找不到合適工種」「照片無法使用」。
最後一項再選「看不清／資訊不足」或「非施作照／不適用」，後端分別記 `uncertain`、`out_of_scope`；
前兩項分別記 `accept`、`correct`。即使模型信心高，也保留「找不到合適工種」，避免強迫使用者選錯類。
`selectedClass` 必須在該版分類表；「找不到合適工種」記 `new_candidate`，只建立候選事件，
不由現場自由字串直接創類或改模型。未出現在模型輸出、但已在分類表的工種仍走 `correct`，
並標記 `known_untrained`，不能重複建立新類。
`evidenceBoxes` 使用現有 0–1000 相對座標契約。API 從 manifest 回查案場、日報日期、照片雜湊與來源，
拒絕非有效 `WORK_ITEM`、壞圖與未知的 `modelVersion`／`catalogVersion`；歷史版本仍可回報，複核時對照當時分類表。
伺服器保存接收時間、
actorId、角色、事件 ID 與原始預測。重送同一 key 回同一事件，不覆寫舊事件。

新工種候選另用 `POST /v1/pms/class-candidates`，最少帶暫定名稱、代表照片 `fileIds`、可見共同特徵與
判斷依據；可補充不屬於此類的照片或情況。先讓現場在照片頁按「找不到合適工種」並留下描述，
品管再彙整多張代表照片、查重既有類別與規範，交專家核准正式名稱、定義、排除條件與分類樹／QS 依據。
既有 `core/pms_review.py` 已有 `propose_candidate()` 與 `resolve_candidate()`，可沿用同樣的狀態規則。
核准只更新分類表；候選照片仍須逐張人工裁決，累積足夠且跨案場的樣本後才進下一版重訓和評估。
在那之前，系統顯示「待審新工種」，不可硬塞到最像的舊類或宣稱模型已學會。

專家端用 `POST /v1/pms/photo-feedback/{eventId}/review` 提交 `confirm`／`reject`／`needs_adjudication`，
保存 reviewerId、依據與先前事件版本。分歧時另產生仲裁事件。所有步驟追加紀錄，不用單欄位覆蓋歷史。

## 回訓邊界

```text
PMS／EagleField 照片頁 → 預測 API（模型版本、信心、類別）
                         ↓
             現場接受／改判／新類候選／無法使用事件
                         ↓
                    品管複核／仲裁
                         ↓
               已確認訓練快照（不可變）
                         ↓
       前處理 → 特徵 → 案場×日期切分 → 訓練 → 固定黃金集 G2
                         ↓
                人工核准切換／可回退版本
```

現有 `core/pms_review.py` 的 `decide()`／`data/review.csv` 是人工真值入口，
`core/pms_store.py` 已有追加事件與版本檢查；API 應在共用服務層接入這些規則，
不要讓現場 App 直接寫 CSV 或 SQLite。因現有 CSV 與 SQLite 不是跨檔案交易，
正式多人併發服務需要把真值和事件放到同一個交易資料庫，再從快照匯出給現有訓練程式。

黃金考卷須是獨立 PMS 照片，於訓練前按「案場 × 日期」整組保留；
現有 G1 工種抽樣以 legacy 為主，且目前尚無仲裁答案，不可用來判定 v39／v40 上線。
現場的 `accept` 可能只是順手點選，不能等同專家確認；只有經複核的事件進訓練，
另把接受率、改判率、資訊不足比例按案場／工種／模型版本監測，作為重訓觸發訊號。

## 落地順序

1. 在 PMS／EagleField 的照片頁增加四個主要操作：「正確／改成既有工種／找不到合適工種／照片無法使用」；最後一項要求選原因。新類候選先收描述與代表照片，供品管合併審查。
2. 建伺服器端回饋 API 與事件表，串既有登入、角色、冪等鍵和原圖來源驗證；先只收資料。
3. 品管複核佇列與仲裁，完成已確認事件的訓練快照匯出；保留完整來源追溯。
4. 收集獨立 PMS 黃金集、雙人標註與仲裁，重訓兩個都未見過黃金日期的比較版本。
5. 接重訓排程、G2 檢查、影子模式、人工切換與回退；先不讓 API 即時改模型權重。

## PMS 日報測試準入判斷（2026-09-25）

`CURRENT` 仍是 v39（含 legacy 訓練資料），v40 是已完成離線訓練的 PMS-only 候選版，未切換。
v39 的單次測試為 top-1 0.7463、macro-F1 0.6888；v40 為 0.8521、0.8277；
兩版測試照片不同，數字不可直接當作同卷提升幅度或產品正確率。現有 G1 以 legacy 為主，
獨立 PMS G2 黃金集尚未完成。模型的輸出也不能取代品管裁決。

本機已具備 `src/pms_inference.py` 的 v40 唯讀推論，以及 `src/pms_shadow_api.py` 的 loopback HTTP
影子入口。它只接受已同步 manifest 的有效 `WORK_ITEM` `fileId`，從 PMS 原圖重做相同遮蔽、JPEG 編碼、
SigLIP 特徵與探針推論，不寫入正式工種或人工答案。現有 `src/predict.py` 的 ONNX 路徑服務的是另一種
finetune 模型，不能拿來代替 v40。對一張已知照片，原圖即時前處理與批次前處理位元組完全相同，
即時與快取 embedding 最大差 `1.2e-7`，預測類別及六位小數分數一致。

```bash
uv run --extra train python src/pms_shadow_api.py --model v40
curl -s http://127.0.0.1:8765/v1/pms/photo-predictions \
  -H 'Content-Type: application/json' -d '{"fileId":"已同步的 PMS fileId"}'
```

成功回應包含 `status=review_required`、`suggestedClass`、未校準的 `modelScore`、前三個候選、
`modelVersion`、`modelSha256`、`catalogVersion`、`encoderVersion`；壞圖或非 WORK_ITEM 回錯誤碼。
`modelScore` 不能直接當「有九成把握」或自動採用門檻；目前沒有經獨立資料驗證的未知類／拒答閾值，
新工種必須由現場按「找不到合適工種」提出。日報新增照片須先同步到 manifest 與原圖目錄，API 才可讀。
這個本機入口只綁 `127.0.0.1`，沒有 PMS 使用者驗證；正式跨主機服務仍須接 PMS 後端身分、授權、
服務監控與部署機制。評估影子模式成效時只計算模型訓練後新收集、由人核對的照片，不混入訓練照片。

第一階段可先做限量的唯讀影子測試：在 PMS 日報中記錄預測、耗時與錯誤，預設不寫回正式工種，
也不把現場按「正確」直接當真值。若要讓使用者看到建議，須明確標示「測試建議」，
由人確認後才採用，並留原本人工流程及回退開關。正式自動帶入或據此回訓，
須等獨立 G2、專家複核和監控／回退流程完成。

## 工項批次推論與缺失旗標（2026-09-28，v43）

**模型**：v43 = so400m 編碼器（`ViT-SO400M-14-SigLIP-384`）＋ labels.yaml v17。同一批 338 張考卷上，
單張 top-1 0.8935、工項融合 0.9467，與 v42 無顯著差異；v42 起的 PMS-only 版本才適用下列欄位。
所有回應 `status` 一律 `review_required`：G2 獨立黃金集完成前不自動採用任何建議。

**`POST /v1/predict-batch`**（獨立服務，`X-Service-Token` 驗證）：一次送同一施作項目
（同日報 × 同標題）的全部照片，服務回每張單張結果與工項融合結果。原 `/v1/predict` 單張端點保留、欄位不變。

```json
{
  "workItemId": "PMS 施作項目 id（冪等對帳用）",
  "sourcePolicy": "clean-raw-v1",
  "title": "5F內部牆面粉光（選填；打底/粉光分層判斷會用）",
  "photos": [
    {"fileId": "PMS-fileId-1", "contentType": "image/jpeg", "dataBase64": "..."},
    {"fileId": "PMS-fileId-2", "contentType": "image/webp", "dataBase64": "..."}
  ]
}
```

限制：1–8 張、每張 ≤5 MB、`fileId` 不可重複；任一張壞圖整批回 422（`detail` 帶 fileId）。回應：

```json
{
  "workItemId": "...",
  "status": "review_required",
  "workItem": {"suggestedClass": "泥作-粉光", "modelScore": 0.51, "alternatives": [...],
               "lowConfidence": true, "stageDecision": "title"},
  "photos": [
    {"predictionId": "...", "fileId": "...", "suggestedClass": "泥作-打底", "modelScore": 0.48,
     "alternatives": [...], "modelVersion": "v43", "modelSha256": "...", "catalogVersion": "...",
     "encoderVersion": "so400m/ViT-SO400M-14-SigLIP-384/webli", "preprocessVersion": "clean-raw-v1",
     "fused": {"suggestedClass": "泥作-粉光", "modelScore": 0.5, "alternatives": [...],
               "lowConfidence": true, "stageDecision": "title"}}
  ],
  "modelVersion": "v43",
  "encoderVersion": "so400m/ViT-SO400M-14-SigLIP-384/webli"
}
```

- `photos[].suggestedClass` 是單張判斷；`photos[].fused` 是參考同工項其他照片後的判斷（v42 同卷 +5pt），
  現場建議請顯示 `fused`。`workItem` 是整個工項的綜合結果。每張的 `predictionId` 照舊寫入回饋收件匣，
  `/v1/feedback` 不變。
- `lowConfidence`：融合後信心 < `reviewConfidence`（模型包 metadata，v43 為 0.52；分組 OOF 上
  precision ≥ 0.95 的門檻）或打底／粉光階段待人工。只是分流提示，未經 G2 驗證，不能當自動採用門檻。
- `stageDecision`：`model`＝模型直接判；`title`＝模型只確定是「泥作打底/粉光群」、階段不確定，
  依 `title` 明確寫的打底或粉光決定；`manual`＝標題沒寫階段，請人看圖。不送 `title` 就不會出現 `title`。
- `encoderVersion` 前綴改為模型包的 `encoderKey`（舊包無此欄時為 `siglip`）。

**缺失旗標**（與工種正交）：QS 標準總表 76 份、1,229 檢查項與計價樹 489 節點都沒有「缺失改善」計價項目，
v17 起缺改照照 QS 歸工種，「是不是缺改」改由工作台的缺失旗標表達（標題缺失字或 defect-probe 圖像分數，
人工可在「③照片工種」改）。看不出工種的缺改照暫留 `雜項-缺失改善` 並列入人工看圖。
本機影子入口（`src/pms_shadow_api.py`）另回 `stageDecision`、`defectTitle`、`workItemPhotos`、`lowConfidence`。
