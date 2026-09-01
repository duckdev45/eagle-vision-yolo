# =============================================================================
# File: yolo_preprocessor.py
# Purpose: To manage the lifecycle of Open-Vocabulary Annotations for YOLO training.
# The script emulates the human-in-the-loop process:
# 1. AI Prediction (Grounding DINO/SAM): Generates initial bounding boxes from text prompts.
# 2. Human Correction: Allows for manual refinement/deletion/addition (simulated here).
# 3. Output: Creates structured, clean, and ready-to-train YAML/JSON/TXT datasets.
#
# Architecture Philosophy:
# - This module must not function as the *training* engine, but as the *data generator*.
# - Output MUST be segmented by annotation status: 'Initial_AI_Guess' vs 'Human_Refined'.
# - The final source of truth must be the 'Human_Refined' set.
# =============================================================================

import json
from pathlib import Path
from typing import Any

# --- Configuration ---

# Base Root Path (Should match other module conventions)
# src/yolo_preprocessor.py → parent = src/ → parent.parent = 專案根
ROOT_PATH = Path(__file__).resolve().parent.parent
# Output structure: {dataset}/yolo_dataset/pre_annotations/{fileId}/
YOLO_DATA_ROOT = ROOT_PATH / "data" / "derived" / "yolo" / "pre_annotations"

# Critical Annotation Prompts (Derived directly from QS/Contract/Pain Points)
# These are the "Open Vocabulary" terms that the AI should search for.
ANNOTATION_PROMPTS = [
    # From QS0302 / QS0701 / QS0402 etc.
    "縫隙",
    "裂縫",
    "髒污",
    "縫隙",
    "插座",
    "管道口",  # 通用缺失類
    # Specific from the 7 high-value items (Payment/Claim):
    "模板清洗",
    "牆柱清潔口",
    "開口及角隅",
    "鋼筋直徑板",
    "防水層",
    "玻璃硅脂",
    "黃色標記",  # 某個特徵的顏色提示
]

# --- Abstract/Mock Functions ---


def call_grounding_dino(image_path: Path, prompts: list[str]) -> list[dict[str, Any]]:
    """
    MOCK FUNCTION: Simulates calling the Grounding DINO API/Service.
    In a real implementation:
    1. Calls the API with the image path and prompts.
    2. Receives a list of {bounding_box: [x1, y1, x2, y2], confidence: float, label: str}.
    """
    print(f"--- MOCK API CALL: Calling Grounding DINO on {image_path} with {len(prompts)} prompts ---")
    # Simulate a few successful detections
    if "設備-電梯" in str(image_path):
        return [
            {"box": [100, 200, 200, 300], "confidence": 0.95, "label": "電梯"},
            {"box": [500, 150, 600, 250], "confidence": 0.88, "label": "縫隙"},
        ]
    else:
        return []


def run_semi_automatic_annotation(
    image_path: Path, all_annotations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    CORE LOGIC: Simulates the Human-in-the-Loop (HITL) refinement process.
    """
    print("\n*** Starting Human-in-the-Loop Annotation Refinement ***")

    # 1. InitialAI_Guess: 這些框是機器抓的，需要人工檢查
    initial_guesses = []
    for ann in all_annotations:
        initial_guesses.append(
            {
                "box": ann["box"],
                "confidence": ann["confidence"],
                "label": ann["label"],
                "status": "AI_GUESS",
                "original_source": "GroundingDino",
            }
        )

    # 2. Human_Refinement: 這裡假設用戶手動修正了 2 個框
    # 修正邏輯模擬：
    # - 修正 1: 根據專業知識，將一個 '電梯' 框修正為更準確的 '電梯門' 框。
    human_mock_correction = {
        "box": [110, 210, 190, 290],  # 模擬修正後的座標
        "label": "設備-電梯門",  # 模擬修正後的類別名稱
        "status": "HUMAN_REFINED",
        "manual_note": "根據 QS0302，門型邊界需拉緊。",
        "source": "Manual_Correction",
    }

    # 3. 人工添加/修正一個全新框（例如，發現了機器沒有抓到的細小汙漬）
    human_mock_addition = {
        "box": [700, 50, 720, 70],
        "label": "髒污-小點",
        "status": "HUMAN_ADDED",
        "manual_note": "肉眼可見的細小污染物。",
        "source": "Manual_Addition",
    }

    refined_annotations = [*initial_guesses, human_mock_correction, human_mock_addition]

    return initial_guesses, refined_annotations


# --- Main Processing Pipeline ---


def process_single_file(file_id: str, image_path: Path) -> None:
    """
    Processes a single image file through the AI -> Human -> Output cycle.
    """
    print("\n=========================================================")
    print(f"Processing File ID: {file_id}")

    # Step 1: AI Prediction (Run Grounding DINO)
    all_anns = call_grounding_dino(image_path, ANNOTATION_PROMPTS)

    # Step 2: Human-in-the-Loop Refinement
    _initial_guesses, final_refined = run_semi_automatic_annotation(image_path, all_anns)

    # Step 3: Output Generation (The final deliverable)
    print("\n*** Writing Annotation Outputs ***")

    # For demonstration, we only save one final 'example' file.
    output_dir = YOLO_DATA_ROOT / file_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the final list of annotations that are human-validated
    annotation_output_path = output_dir / "annotations_report.json"
    with open(annotation_output_path, "w", encoding="utf-8") as f:
        json.dump(final_refined, f, ensure_ascii=False, indent=2)

    print(
        f"\n[SUCCESS] Annotation report saved to {annotation_output_path} (Only includes human-verified frames)."
    )

    # NOTE: In a real scenario, you would then iterate through final_refined:
    # for ann in final_refined:
    #     save_yolo_txt_format(ann, output_dir / "yolo_{}_{}.txt", file_id, ann['label']) # {bbox}


def process_dataset(file_ids: list[str], image_dir: Path) -> None:
    """
    Processes a list of file IDs from a dataset subset.
    """
    print("=======================================")
    print(" Starting YOLO Dataset Processing Pipeline ")
    print("=======================================")

    for file_id in file_ids:
        # Assume image extension is always .jpg for this run
        image_path = image_dir / f"{file_id}.jpg"

        if not image_path.exists():
            print(f"[WARNING] Image not found for {file_id}. Skipping.")
            continue

        process_single_file(file_id, image_path)


if __name__ == "__main__":
    # ---- 模擬運行數據 ----
    # 假設我們從一次審核會話中，挑出這兩張圖作為 PoC 訓練樣本
    SAMPLE_FILE_IDS = ["設備-電梯", "真泥作-壁磚貼飾_猜泥作-地磚貼飾_9b3335fc"]
    # 假設這些圖片都放在一個統一的目錄下，模擬從本次審核中收集
    MOCK_IMAGE_DIR = Path("../../../qms_labels.yaml/comparison.md")  # 使用某個資料檔目錄作為模擬圖片來源

    # 實際運行時，這裡應該指向一次QC流程產生的原始照片資料夾。
    # YOLO會從這裡讀取，但由於我無法訪問實際的 QMS_PHOTOS 路徑，這裡先用模擬路徑。

    process_dataset(SAMPLE_FILE_IDS, MOCK_IMAGE_DIR)
