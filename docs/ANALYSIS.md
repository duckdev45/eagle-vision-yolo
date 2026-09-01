# 分析結論彙整（2026-08）

> 取代原 `reference/iso/` 14 份分析筆記與根目錄 4 份 workflow md（已刪，git 可救回）。
> 每節末的〔原 xxx.md〕標示結論出處；統計數字以 `uv run src/qsdata.py` 現算為準。

## 一、資料層：QS 公司標準的結構與價值

**結構**：76 份標準（catalog.yaml 目錄），每份 checklist 兩層——`OPTIONAL` = 工序階段節點（143 項）、`REQUIRED` = 實際檢查項（1,075 項）。主鍵 `(docNo, itemNo)`，如 `QS0404-4` 貼地磚厚度。〔原 ISO_MAPPING / WORKTYPE_PHASE_MAP / ITEM_LEVEL_USES〕
⚠ 舊排查報告（AUDIT_2026-08-28）稱 1,304 項——那是電梯那份從 PDF OCR 重落地**之前**的數字，現檔以 `make qs` 現算為準。

**小項才是價值所在**：大項（OPTIONAL）只有 3/10 核心類有現成工序，且分「時序/部位」兩種語意；小項承載驗收數值、請款照片憑證、缺失分類表。〔原 ITEM_LEVEL_USES〕

**A~E 工具分派實測**（啟發式）：A 純視覺 62% / B 量測 19.3% / C 文件 12.5% / D 時序 4.1% / E 儀器 2.2% → A+B = 81.2% 在 vision 守備範圍。C+D+E 集中在材料檢驗（QS08）與定期檢查，整批延後不影響主線。〔原 FLOW_OPTIMIZATION / DEFECT_ML_PLAN〕
**但注意**：抽 10% A 類人工覆核，部分應標 unjudgeable（如「材料是否良好」無可操作判準）→ 真實 A+B 約 77~80%，勿引用 85%。〔原 DEFECT_ML_PLAN §三〕

**覆蓋陷阱**：`OPTIONAL` 只覆蓋 3/10 核心類，不代表 QS 沒有工序——37 份盤點 A 型（有現成階段）12 份、C 型（扁平）20 份；全庫 75 份則 A 35 / C 41。C 型的階段需從小項順序推論且**未經人工驗證**。〔原 WORKTYPE_PHASE_MAP〕

## 二、合約側：比 QS 更細，且綁定錢

**首批 6 份 / 195 條已落地**（泥作 82 / 防水 41 / 油漆 40 / 灌漿牆 12 / 木作 9 / 鋼筋材料 11）。〔原 CONTRACT_VS_QS〕

**判定以合約為準**，不是因為合約比較嚴，而是 QS 自己把裁量權讓給合約（QS0302-5 明文「依**合約中工作約定要點**施工」）。已人工確認 4 處數值衝突（合約比 QS 嚴：浴室泛水 180cm vs 20cm、磁磚縫 1~2mm vs <3mm、平整度 <1.5mm/6尺 vs <2mm/m、切細料≤5cm QS 沒寫）。合約主鍵**帶案名**，與 QS 分開 index——同一工種在不同案數值不同，混用會拿 A 案條款答 B 案問題。〔原 CONTRACT_VS_QS〕

**油漆施工說明書比 QS 更細**：QS0501 油漆 4 階段 vs 合約 12 種施作面各自完整工序。〔原 CONTRACT_VS_QS §四〕

**鋼筋那份是物明不是工明**（材料供應契約，非施工約定）→ 解不了 QS0302-5 鋼筋綁紮間距。〔原 CONTRACT_VS_QS §五〕

**介面是合約獨有的價值**：QS 按單一工種編排，工種交界被切碎（28 個介面項散在 9 份標準）；合約逐工種簽，同一交界在兩份合約各出現一次，兩邊都拿到才完整。防水工約是首個完整實例（D8 硬性順序閘門只寫在防水合約）。〔原 KNOWLEDGE_OBJECT_ROUTING / CONTRACT_VS_QS §七〕

**97 項合約相依只對應 5 項**：判定基準指向合約的 REQUIRED 項佔 9.0%（97/1,075；舊筆記的 7.3% 是用過時分母 1,304 算的），單灌 QS 進 RAG 這些問題檢索得到卻答不出來。優先序須以 97 項（非舊 24 項）重排。〔原 AUDIT / CONTRACT_PRIORITY〕

## 三、缺失 ML：先分類再選工具

**三段判定性質不同**：工種=ML（已有 0.861）→ 缺失=ML+人（標籤無免費來源）→ 請款=程式 json-logic（可稽核，ML 判請款失去稽核性）。〔原 DEFECT_ML_PLAN / LLM_DEPENDENCY / KNOWLEDGE_OBJECT_ROUTING〕

**缺失按視覺樣態分群**（縫隙/接合 48 項、表面清潔 43 項、成品保護 25 項…），不按 QS 項——每 QS 項一個分類器需 30,000 標註，差 25 倍。缺失與工種是**正交維度**（同一張可同時是「地磚貼飾」和「縫隙不齊」），不可塞進 labels.yaml 的 cls。〔原 DEFECT_ML_PLAN〕

**三層任務分解**：存在性（該拍的有沒有入鏡）→ 品質判斷（做得好不好）→ 流程合規（時序對不對）。最高槓桿是退回按鈕存 QS 代碼。〔原 DEFECT_ML_PLAN〕

**B 類被低估的解法**：QS0701 §4.14「防水高度**以尺丈量**並附相片存證」——標準要的就是照片裡有把尺，判「量測工具入鏡」遠比判「牆平不平」容易。B 類可能大半能降格成 A。〔原 NEXT_PHASE / ITEM_LEVEL_USES〕

## 四、工種辨識模型：現況與封板結論

**工種辨識已無 LLM 依賴**：本地 88.8% vs Gemini 65.4%（同一份卷）。但有三個誠實折扣：黃金集只 112 張（±8pt）、門檻用測試集算（樂觀）、新工地新角度持續產生低信心樣本——混合架構（本地+fallback）是穩態不是過渡。〔原 LLM_DEPENDENCY〕

**混淆集中在陰陽角收邊**：對 12/12 run 混淆的油漆↔泥作誤判照逐張驗證（3 張成功），誤差集中在「陰角收邊+小工具點狀動作」，且該子動作在兩份標準都有明文檢查項（QS0402 §7/§12、QS0501 §2/§3）。另發現：部分誤判實為**標籤錯誤**（泥作師傅代工油漆前置，模型猜對了工序畫面、錯的是 ground truth）。〔原 PHASE_VERIFICATION_01〕

**已封板**：QMS 36k 張對日報有害（混訓/純遷移/預訓練全試過，領域偏移）；微調編碼器毀掉診斷性；不取代 Gemini，用信心門檻串接。測試集 n=108 時 1 張=0.93pt，單點分數不可下結論，看分組 CV。〔原 WORKFLOW §四、五〕

**QMS 稽核照分類器（0.884，n=2204）是獨立產品線**，不是日報實驗的失敗品。〔原 WORKFLOW §四〕

## 五、規範→判定鏈（LLM 的正確位置）

「完全不依賴 LLM」的正確範圍是**判定鏈上沒有 LLM 當法官**，不是系統裡沒有 LLM：
- L2 規則編譯（中文規則→JSON）、L6 理由生成（結構化→人話）——語言轉換，不在判定路徑上，不該取代
- 工種 Gate 已由本地探針取代；L5 請款判定是 json-logic；「LLM 當翻譯官不當法官」〔原 LLM_DEPENDENCY〕

合約知識包的欄位歸屬：規定類→RAG；缺失案例照→**詞彙表不是訓練集**（負樣本有標籤但量小）；Hold Point/付款比例/罰則→L5 規則引擎輸入。〔原 KNOWLEDGE_OBJECT_ROUTING〕

## 六、資料層守護（2026-08-28 排查後的三道防護）

1. `docNo` 衝突不再靜默解決：QS0907=電梯、QS0907B=停車場排風——本專案固定編號，勿改回
2. 自檢納入 `make test`：21 項測試 + qsdata 11 項 + contractdata 11 項自檢
3. 爬蟲寫入前先跑一致性檢查（fetch_iso_checklist.py）〔原 AUDIT〕

QS0101 不在目錄但被 QS0105 §3.2 引用 → 待品管部確認（當作不存在 vs 停用，意義不同）。

## 七、被作廢的結論（勿再引用）

| 作廢內容 | 原因 |
|---|---|
| 「A+B = 85.1%」 | 樣本 603 項時算的，全庫 1,304 項後為 81.2%；且含 unjudgeable 誤判 |
| 「大項覆蓋 3/10」適用全域 | 只在核心 10 類成立；37 份裡 A 型 12 份（32%） |
| 通用知識生成的工序數值 | 2026-08 搜尋工具故障期自產，與 QS 三處衝突（地磚厚度/縫寬/粉刷平整度）——一律以 QS 原文為準 |
| CONSTRUCTION_WORKFLOW.md 等三份 workflow md 的數值 | 自標「已被取代、數值不可引用」 |
| CONTRACT_PRIORITY.md 的 24 項優先序 | 合約相依已從 24→95→97 項，需重排 |

## 八、檔案對照（git 留底可救回）

| 原檔 | 結論去向 |
|---|---|
| ISO_MAPPING / WORKTYPE_PHASE_MAP / ITEM_LEVEL_USES | §一 |
| FLOW_OPTIMIZATION / DEFECT_ML_PLAN | §一、三 |
| LLM_DEPENDENCY / KNOWLEDGE_OBJECT_ROUTING | §五 |
| PHASE_VERIFICATION_01 | §四 |
| CONTRACT_PRIORITY / AUDIT / COMPLETION_REPORT / ISO_STATUS / FETCH_INSTRUCTIONS / OVERVIEW | §二、六 + `reference/iso/README.md` |
| CONSTRUCTION_WORKFLOW / DETAILED_WORKFLOW_ANALYSIS / WORKFLOW_SUMMARY / PROJECT_OVERVIEW / ARCHITECTURE_DESIGN | §七 + `README.md` |
| NEXT_PHASE | → `docs/ROADMAP.md` |
| WORKFLOW.md（訓練工作流） | 保留：流程契約，內容仍有效 |
