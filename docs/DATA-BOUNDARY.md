# 資料邊界：什麼可以外售，什麼絕對不行

> 2026-10-09 立案。公司級定位改成「賣經過營建專家審核的判斷」之後，這張紙比技術文件更急——
> 對客戶承諾的是「**照片不出內網**」，對外賣的是「**判準**」，兩件事必須明確切開；
> 模糊一次，信任就沒了。
>
> **這份文件不是宣言，是程式的白名單。** `src/export_label_pack.py` 會解析下方 `boundary-spec`
> 區塊，只匯出這裡列到的欄位、只接受這裡宣告的型別，並把本檔的 sha256 寫進判準包的
> `manifest.json`。改了這份文件，匯出的東西就跟著改；刪了這份文件，匯出直接拒跑。

## 一、可外售（判準包的內容）

| 類別 | 具體內容 | 為什麼可以 |
|---|---|---|
| **人審裁決紀錄** | 「規則/模型說 A，懂現場的人改成 B」這組對照 | 這是我們自己產生的判斷，不是客戶的營運事實。Surge 賣的就是這一段 |
| **工種分類樹（我們的）** | `labels.yaml` 的**類別名單**與版本號 | 類別名是我們定的標籤體系；注意**規則 pattern 本身不外售**（裡面是日報用語語料） |
| **缺失樣態** | 10 種樣態 enum（縫隙/收邊、髒污/殘留…）與框座標 | 樣態是我們歸納的；框是幾何，不帶身分 |
| **QS 判準的結構** | 條號（`QS0403-4.1`）、A~E 工具分派、O/R、請款靶/罰則/合約相依三個旗標 | 條號是路由鍵、分派是我們的分析成果 |
| **統計與一致性** | 類別分佈、標註者一致度、人審改寫率 | 聚合數字，不可回推個案 |

## 二、不可外售（紅線）

| 不可 | 原因 |
|---|---|
| **原始工地照片、任何照片衍生檔** | 對客戶的承諾。判準包裡一個位元的影像都不會有 |
| **公司品質標準／合約的原文**（條文句子、查驗項名稱、工序總表） | 那是公司（與業主）的文件內容，不是我們的分析成果。條號可以、內容不行 |
| **案場名稱、`constrId`、日報編號** | 客戶的營運事實。案場只以 `site-001` 這種包內序號出現 |
| **人名**（`createdBy`、`reviewedBy`、標註者本名） | 個資 |
| **自由文字**（`title`、`note`、`location`、`chipsOn`、`specKey`） | 工地主任手打的字，含樓層、戶別、廠商、案場簡稱——去識別化無法靠人工眼睛保證，所以整欄不給 |
| **`fileId`** | 它是客戶 PMS 的主鍵。給了等於給一把回頭對照原始照片的鑰匙，即使照片沒出去。包內一律用加鹽雜湊 `sampleId` |

### 為什麼連 `fileId` 都要雜湊

一份包裡有 `fileId` 的話，只要買方日後與客戶有任何接觸（或我們自己外流一份 manifest），
兩邊 join 就還原出「哪個案場的哪張照片被判成什麼」。加鹽雜湊讓**同一包內可以 join**
（買方照樣能把裁決與框對起來），但**包外無法回推**。鹽與對照表留在本機 sidecar
（`<pack>.local.json`，0600，不進包、不進 git），只有我們自己能追溯。

## 三、匯出紀律

1. **白名單制，不是黑名單**：新欄位預設**不**外售。想加欄位 → 先改這份文件，再跑匯出。
2. **只收人審**：沒有人審裁決的照片不進包（與 YOLO 訓練集同一條紀律，`docs/ROADMAP.md`）。
3. **型別即防線**：每個欄位宣告型別，值不符就**整份拒絕匯出**（fail loud，不靜默清洗）。
   這是為了擋住「某天有人把 `note` 塞進 `defectType`」這種事。
4. **包要能被稽核**：`manifest.json` 記 schema 版本、本檔 sha256、每個檔案的 sha256 與列數。
5. **授權未簽，不得出包**：資料授權條款（誰擁有照片、去識別化標註可否用於訓練與聚合外售）
   簽核前，判準包只能在內部使用。匯出時用 `--licensed` 明示已簽，否則包會標成
   `distribution: internal-only`。
6. **產出位置**：`data/exports/label-packs/<version>/`（`data/` 已 gitignore）。
   回推對照表是**同層的** `<version>.local.json`，刻意放在包的外面——
   `zip -r pack.zip <version>/` 不會把它一起帶走。

---

## boundary-spec

<!-- 以下 yaml 由 src/export_label_pack.py 解析；改欄位請連同上方表格一起改。 -->

```yaml
schemaVersion: 1
types:
  hash16: salted sha256 前 16 碼（包內 join 用，包外不可回推）
  bucket: 包內案場序號 site-NNN
  yearMonth: YYYY-MM（不給日，日＋案場足以回推單一日報）
  taxonomy: 必須是 labels.yaml 的類別或已核准新類
  defectPattern: core.defects 的樣態 enum
  qsCode: QSNNNN 或 QSNNNN-x.y（條號可外售，名稱不可）
  box1000: 四個 0~1000 整數（相對原圖）
  token: 小寫英數與連字號，長度 ≤32（擋中文自由文字）
  flag: 0 或 1
tables:
  judgements:
    desc: 照片層級的人審裁決與「人改了機器什麼」
    columns:
      sampleId: hash16
      siteBucket: bucket
      yearMonth: yearMonth
      humanClass: taxonomy
      ruleClass: taxonomy
      humanOverrode: flag
      hasBox: flag
  defect_boxes:
    desc: 人框的缺失框（CVAT／退回按鈕），一行一框
    columns:
      sampleId: hash16
      defectPattern: defectPattern
      qsCode: qsCode
      box: box1000
      source: token
  qs_criteria:
    desc: QS 判準的結構（條號與分派，無標準原文）
    columns:
      qsKey: qsCode
      kind: token
      status: token
      isBilling: flag
      isPenalty: flag
      isContract: flag
forbiddenColumns:
  - fileId
  - title
  - note
  - location
  - chipsOn
  - chipsCustom
  - specKey
  - constrId
  - constrName
  - site
  - siteRaw
  - createdBy
  - reviewedBy
  - annotator
  - dailyReportInfoId
  - objectKey
  - fileName
  - name
  - reportDate
  - reviewedAt
```
