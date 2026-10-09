# 待審清單契約（給未來的標註平台）

> 2026-10-09 立。定案方向：之後會有一個平台給懂的人去標，標完經 API 寫回。
> 這份文件定的是**兩邊的介面**；平台本身與 API 伺服器還沒做。改欄位＝先改這份文件，再改 `core/routing.py`。

## 一、匯出：`data/exports/queue/queue.json`

每次分流（`make daily`／`make route`）覆寫一次。內容是公司資料（含日報標題、fileId），
只放在 gitignored 的 `data/` 底下，**不是**對外交付物——對外交付走判準包（`docs/DATA-BOUNDARY.md`）。

```json
{
  "schemaVersion": 1,
  "generatedAt": "ISO-8601 UTC",
  "model": "現行模型版本",
  "autoConfidence": 0.9,
  "catalog": ["完整分類表的類名，平台的下拉選單只能從這裡選"],
  "items": [
    {
      "fileId": "PMS 檔案 id",
      "bucket": "queue | audit",
      "reasons": ["為什麼要人看，見下表"],
      "ruleClass": "純標題規則的類別，可能空字串",
      "modelClass": "模型類別，可能空字串",
      "modelConfidence": 0.0,
      "title": "日報標題",
      "reportDate": "YYYY-MM-DD"
    }
  ]
}
```

`items` 已依優先序排好（規則與模型吵架的在最前）。`reasons` 的值只會是：

| 值 | 意思 |
|---|---|
| 規則與模型不同 | 兩個訊號給不同答案 |
| 標題沒有對應規則 | 標題對不上任何規則 |
| 模型尚未涵蓋此類 | 規則給的類別模型沒學過（樣本不足） |
| 打底／粉光階段待人工 | 圖像判是打底粉光群，標題分不出階段 |
| 缺失改善照：工種待看圖 | 依定義要看圖才知道是哪個工種 |
| 模型信心不足 | 規則＝模型但信心 < `autoConfidence` |
| 自動桶抽查 | 本來會自動確認，固定抽 5% 給人看（**標註者不該知道這張是抽查**，平台介面不要顯示這個理由） |

照片本體不在 JSON 裡；平台用 `fileId` 向內網取圖（照片不出內網）。

## 二、寫回：唯一入口 `core.routing.resolve()`

```python
routing.resolve(file_id, reviewer="標註者帳號", label="類名")  # 分類
routing.resolve(file_id, reviewer="…", action="uncertain", reason="看不出來的原因")
routing.resolve(file_id, reviewer="…", action="excluded", reason="不該分類的原因")
```

它走工作台同一條 `pms_decisions.decide()`：驗證類名在分類表裡、要求確認者、暫緩／排除要寫原因，
同時寫 `data/review.csv` 與 `data/pms_review.sqlite3` 的事件紀錄。**API 層不得另開寫入路徑。**
寫回後該照片立刻離開收件匣，不必等下次分流。

## 三、API 伺服器要補的（還沒做）

- 認證：`reviewer` 必須來自登入身分，不能讓呼叫端自填。
- 並行：現在是單機 CSV＋SQLite，不是交易一致的多人服務（`PMS-CLASSIFICATION.md`「儲存與模組」）。
  多人同時標之前，寫回要排成單一寫入者（佇列或鎖），或先把 `review.csv` 收進 SQLite。
- 衝突：同一張被兩個人標不同答案時的仲裁規則（G1 的雙標仲裁可沿用）。
- 抽查桶的答案要能拿來算 `routing.audit_precision()`——寫回時不需要特別處理，只要照常寫人審即可。
