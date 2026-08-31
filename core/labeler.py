# core/labeler.py
"""
標籤推理引擎 (Labeling Engine)。
包含所有的規則匹配、分類邏輯，負責將文字標題轉換為標準化的類別代碼 (Labbels)。
這是核心的業務規則層 (Business Rule Layer)，必須保持高度隔離。

---
【語言規範提醒】
1. 程式碼和註解：只使用 English 和 Chinese。
2. 核心資料結構：應由 core/models.py 定義。
3. 寫入與讀取邏輯：純邏輯，不包含 Streamlit 執行介面。
---
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Import core models to keep everything clean
from core.models import CoreItem, BaseArtifact

# --- 核心配置 (Configuration) ---

# 這是所有規則的集合，順序決定了匹配的優先級（先匹配的為最優先的）。
RULES: List[Tuple[re.Pattern, str, str]] = [
    # (Regex Pattern, Label Name, Description)
    (re.compile(r"試驗|抗壓|氯離子|坍度|扭力|水壓|蓄水|非破壞|試水|含水率|強度.*試|試.*強度"), 
     "Equipment Test", "儀器試驗（工程指標）"), # Type E
    (re.compile(r"證明|證照|報告|圖說|樣品|簽認|計劃書|計畫書|名冊|技術資料|合約|採發|規範|出廠|CNS|認可|核准|審核|存查|備查|提送"), 
     "Document Proof", "文件憑證（文件類）"), # Type C
    (re.compile(r"每月|每半月|每日|天後|小時|分鐘|日內|週內|一週|之前|先行|完成後|時機|同時|再行|方可|後方|次數|定期"), 
     "Time Sequence", "時序流程（流程類）"), # Type D
    (re.compile(r"\\d+\s*(?:mm|cm|m²|m2|kg|µ|%|度|公分|公尺|米|倍|分|寸)\\s*.*"), 
     "Measurable Value", "量測數值（工藝指標）"), # Type B
]
UNSURE = "A"  # If it doesn't match any rule -> Pure Visual (最可能是 VLM 處理的)

# 業務規則的標籤和它們的描述，方便未來擴展
FALLBACK_LABEL = "Other/Unspecified"
MIN_TITLE_LENGTH = 3 # Minimum required title character length

# 靜態載入的排除清單（應從 YAML 或其他配置文件載入）
JUNK_PATTERNS: List[re.Pattern] = [
    # 範例: 增加任何系統不需要考慮的內容模式
]
JUNK_SEARCH_TERMS: List[str] = []
EXCLUDE_DATES: set = set()
EXCLUDE_CREATED_BY: set = set()
EXCLUDE_STATUS: set = set()
IS_FUTURE_DATE: bool = True

class Labeler:
    """
    核心標籤推理器。
    負責將原始標題字串(Title)轉換為標準化的類別代號(Category)。
    """
    def __init__(self, min_len: int = MIN_TITLE_LENGTH, fallback: str = FALLBACK_LABEL):
        self.min_len = min_len
        self.fallback = fallback
        self.rules = RULES
        self.junk_patterns = JUNK_PATTERNS
        self.junk_terms = JUNK_SEARCH_TERMS
        self.min_class_size = 10 # 建議調整為最小值
        self.small_class_policy = "merge" # drop / merge

    @classmethod
    def load(cls, config_path: str = "config/labels_config.yaml") -> Labeler:
        """
        從配置檔載入 Labeler 實例 (For Persistence/Loading).
        實際生產環境應透過配置系統注入這些參數。
        """
        # 這裡假設從某個地方載入了配置，為演示，我們使用預設的 Labeler 建構子
        return cls(MIN_TITLE_LENGTH)
    
    @staticmethod
    def _normalize_title(title: Optional[str]) -> str:
        """
        Standardize the input title. Cleans up NaN, None, and whitespace.
        """
        if title is None or (isinstance(title, float) and title != title):
            return "" # Represents NaN
        
        t = str(title).strip()
        return t
    
    def label(self, title: Optional[str]) -> Optional[str]:
        """
        將輸入標題 (Title) 標籤化。規則按順序匹配，第一個匹配成功即為結果。
        :param title: 原始標題字串。
        :return: 命中的類別名稱，或 None (排除)。
        """
        # 1. 清理與基礎檢查
        t = self._normalize_title(title)
        if not t or t in self.junk_terms or len(t) < self.min_len:
            return None
        
        # 2. 排除檢查
        if any(p.search(t) for p in self.junk_patterns):
            return None
        
        # 3. 規則匹配 (Order Matters)
        for pattern, label, _ in self.rules:
            if pattern.search(t):
                # 匹配上規則，返回規則名稱即可
                return label
        
        # 4. 預設回退
        return self.fallback 

# ------------------------------------------------------------------------
# ** 業務邏輯方法集 (Workflow Methods) **
# 這些方法應被移到上層的 pipeline/ 或 services/ 模組中調用。
# 這裡保留它們，作為一個 API 接口。
# ------------------------------------------------------------------------

def apply_labeling(df: pd.DataFrame, drop_small: bool = True, report_date_today: Optional[str] = None) -> pd.DataFrame:
    """
    對整個 DataFrame 的標題欄位應用標籤邏輯。
    :param df: 包含 'title' 欄位 DataFrame。
    :param drop_small: 是否執行最小類別大小篩選。
    :param report_date_today: (Optional) 當天報告，用於日期排除。
    :return: 成功標籤化後的 DataFrame (增加了 'cls' 欄位)。
    """
    labeler = Labeler()
    
    # 步驟 1: 執行報告層級的排除 (Date/Status check)
    # 這裡需要根據哪個來源的 DF 來判斷，目前是通用版，假設df已經過濾了結構錯誤的資料。
    # 實際應用時，應在調用 get_manifest() 之前呼叫，並傳入過濾後的 DF
    # filtered_df = filter_by_exclusion(df, report_date_today)
    # 簡化為直接在 apply 函式內控制。
    
    # 步驟 2: 標籤化
    out = df.copy()
    out['cls'] = out['title'].apply(lambda t: labeler.label(t))

    # 步驟 3: 移除 Labeler 不應該介入的數據層級 (e.g., If raw data is not the title, label it as None)
    # 這是保持數據純粹性的重要步驟。

    # 步驟 4: 處理小類別 (Min Class Size Check)
    if drop_small and labeler.min_class_size > 0:
        n = out['cls'].value_counts()
        small_classes = set(n[n < labeler.min_class_size].index)
        
        if labeler.small_class_policy == "drop":
            out = out[~out['cls'].isin(small_classes)]
        else: # merge
            out.loc[out['cls'].isin(small_classes - {labeler.fallback}), 'cls'] = labeler.fallback
            
    return out

def get_pending_classes(df: DataFrame) -> DataFrame:
    """
    找出規則已標籤，但數量不足以進入訓練集的類別列表 (Pending Classes)。
    """
    labeler = Labeler()
    # ... (略去複雜的數據處理邏輯，保持原邏輯)
    print("--- get_pending_classes logic placeholder ---")
    return pd.DataFrame()

def get_unclaimed(df: DataFrame) -> DataFrame:
    """
    找出沒有任何規則能標籤（Label）的剩餘資料，這些是潛在新工種。
    """
    labeler = Labeler()
    print("--- get_unclaimed logic placeholder ---")
    # (此處會走 labeler.label() 返回 None 的資料)
    return pd.DataFrame()

# 確保這些核心工具定義能夠被上層的工作流(pipeline/)調用
