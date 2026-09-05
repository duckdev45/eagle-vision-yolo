# 退回按鈕存缺失碼 + LINE 收集契約（2026-09-05）

> ROADMAP 執行序 #2。交 EagleField App owner 的一頁需求：只要**欄位**，不要邏輯。
> 目的：缺失標籤零成本累積。今天不存，一年後是 0 筆（時間不可逆）。

## 1. 退回按鈕（EagleField App / 日報審核）

主任按「退回」時多兩個選單，其餘流程不變：

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| `defectType` | enum | **是** | 10 個視覺樣態（下表）＋ `其他` ＋ `不確定`。單選，預設空白不預選 |
| `qsCode` | string | 否 | QS 項代碼（如 `QS0402-7`），從 catalog 搜尋選取，可空 |
| `note` | string | 否 | 自由文字，現有欄位沿用 |
| `rejectedBy` / `rejectedAt` | 既有 | — | 既有欄位，不動 |

樣態 enum（與 `data/field_reports/README.md` 樣態掃描一致，與工種正交）：

```
縫隙/收邊 · 髒污/殘留 · 破損/脫落 · 滲水/水痕 · 不平整 · 保護不足 · 刮傷/撞痕 · 鏽蝕 · 裂縫 · 掉漆/漆面
```

**選單順序＝樣態在前、QS 代碼在後。** QS 代碼難選，樣態 3 秒可選；樣態必填才有標籤，QS 碼是加分。
帶碼率（有選樣態 ÷ 全部退回）是本軸 North Star proxy，目標 ≥ 70%；低於 50% 代表選單設計錯，回頭改選單不加訓練。

## 2. 落地契約：`data/defects.csv`

退回與 LINE 兩軌寫同一張表，`source` 分流。與 `data/review.csv`（工種 cls 裁決）**分開**——缺失與工種是正交維度。

```
fileId,source,site,location,trade,defectType,qsCode,note,reviewedBy,reviewedAt,box
```

| 欄 | 退回軌 | LINE 軌 |
|---|---|---|
| `source` | `app` | `line` |
| `site` | 日報 constrId | 選建案 |
| `location` | 日報 anno.loc | 選位置（選單由 `anno.loc` 33 個空間詞生成，不手打） |
| `trade` | 日報 cls | 選工種（選單由 `labels.yaml` 生成） |
| `defectType` / `qsCode` | 主任選 | 空（LINE 流程沒有此步） |
| `reviewedBy` | 主任 | 名字/email |
| `box` | 空 | 空（框只從 CVAT 來） |

規則：
- 照片本體進 `data/raw/photos/`（LINE 軌）或既有 fileId（退回軌）；**不准第五源孤兒**。
- ETL 每日一次 append-only，`fileId+source` 去重。
- status 一律 HUMAN 層（人選的），可直接進 G1 分層抽樣池；AI_GUESS 不寫這張表。

## 3. LINE 軌 9/18 前必做：原圖 vs 壓縮判定

LINE 預設重新編碼照片（EXIF 剝除、長邊縮小）。同一張照片兩軌各傳一次，比尺寸與 EXIF：

```
uv run python -c "from PIL import Image;import sys;im=Image.open(sys.argv[1]);print(im.size,'exif' if im.getexif() else 'NO-EXIF')" 照片路徑
```

| 結果 | LINE 軌角色 |
|---|---|
| 尺寸相同、有 EXIF | 與 App 軌同級，可入 det 語料 |
| 縮小或無 EXIF | **只當工種 cls 語料**，不入缺失框語料；D 類時序不可用 |

9/18 前決定，不要事後補。若要原圖，bot 說明文字寫「請用『原始畫質』傳送」並在 ETL 記 `natW/natH` 供事後過濾。

## 4. 不做

- 不做缺失單/複核佇列 UI——沒有 AI_GUESS 流可排。
- 不做 QS 代碼自動推薦——先量帶碼率。
- 不改 `review.csv` 契約。
