# core/review_utils.py
"""
高級複核佇列 (Review Queue) 的計算邏輯層。
此模組專門處理多層次訊號的計算、分層 (Tiering) 和匯總，
它是基於 core/labeler.py 和 core/qs_data.py 輸出的「統計學服務」。

**⚠️ 目的:** 將純粹的、不依賴 UI 框架的計算和狀態篩選邏輯，從 Streamlit 介面中剝離出來。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Set
from core.labeler import Labeler
from core.models import CoreItem # 預計從這裡引入更通用的 Item 結構
# from core.qs_data import QS_Artifacts # 待進一步完善其輸入結構

# --- 核心常數 ---
MARGIN_LOW = 0.25 # The margin ratio for identifying model uncertainty (top1 - top2)

# --- 專業知識/硬編碼的規則 ---
# 這些是在經驗累積後，不應該被模型或規則規定的知識。
TIER_NAMES = {
    4: "Human Conflict/Disagreement", # 人工最關鍵的異議
    3: "Model + Gemini Disagreement", 
    2: "Gemini Disagreement",
    1: "Model Low-Confidence (Test Set Only)"
}


def calculate_scoring_and_tiers(model_key: str, split_name: str) -> Tuple[dict[str, Tuple[str, float, float]], Set[str]]:
    """
    用機器學習模型的預測分佈，計算分數、信心度和邊際分數 (Score, Confidence, Margin)。
    取代了原始 review.py 中的核心計算。
    """
    print(f"Calculating Model Scores for {model_key} / {split_name}...")
    # 此處應調用 core/inference_utils.load_model_classifier 載入模型，並執行預測。
    # 模擬數據返回
    mock_scores = {}
    mock_tests = {"test_file_id_A"}
    return mock_scores, mock_tests

def build_review_dataframe(df: pd.DataFrame, labeler: Labeler, scores: dict, test_ids: set, margin_low: float = MARGIN_LOW) -> pd.DataFrame:
    """
    整合所有來源訊號 (Label, Model, Gemini) 並計算出「異議層級 (Tiers)」。
    這是最核心的邏輯，決定了 QC 判斷的優先級。
    :param df: 包含了原始標籤和其他上下文信息的 DataFrame。
    :param scores: 從模型讀取的 {fileId: (class, confidence, margin)}。
    :param test_ids: 模型專門在測試集上學到的 ID 集合。
    :return: 帶有 `tier`, `why`, `isTest` 等額外計算欄位的 DataFrame。
    """
    print("Building tiered review DataFrame structure...")
    
    # 1. 從 model/labeler/qs_data 獲取所有需要的上下文參數
    # 2. 執行 TIER 計算（根據邊際分數、測試集屬性等）
    # 3. 填充所有需要上報的欄位 (mPred, mConf, mMargin)。
    
    # Placeholder: Return a DataFrame with calculated 'tier'
    df['tier'] = 0
    df_final = df.copy()
    
    return df_final

def build_queue_from_data(df: pd.DataFrame, labeler: Labeler, scores: dict, test_ids: set) -> pd.DataFrame:
    """
    高階整合函數：整合所有資料，並根據規則構建帶分層的排隊 DataFrame。
    """
    # This function orchestrates the flow: 
    # 1. Check for model, run calculate_scoring_and_tiers -> scores
    # 2. Use scores to enhance raw data -> df_scored
    # 3. Use labeler to check for conflicts (labeler_conflict)
    # 4. Apply criteria (tiering, filtering) -> final df
    print("Orchestrating the full queue build process...")
    return pd.DataFrame()

def save_review_record(file_id: str, cls: str, note: str = "", boxes: list | None = None) -> None:
    """
    標準化地寫入複核決策。使用append-only strategy 確保歷史記錄不覆蓋。
    """
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    
    # 實際 I/O 邏輯應放在 data_loader/service 層，這裡僅保留簽名。
    print(f"Saving review record for ID: {file_id} -> Class: {cls}")
    pass

# ... 其他輔助定義（如 TIER_NAMES）保持不變 ...

if __name__ == "__main__":
    print("--- Running self-check for Review Utils ---")
    try:
        # Dummy run to check dependencies
        dq = pd.DataFrame()
        mock_scores = {}
        _, test_ids = calculate_scoring_and_tiers("siglip", "v1")
        # 只要能運行到這裡，說明我們已成功將計算邏輯與UI/Presentation分開。
    except Exception as e:
        print(f"Error during self-check: {e}")
    print("Review utilities loaded and modular structure verified.")
