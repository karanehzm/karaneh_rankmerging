from pathlib import Path


# DATASET_NAME = "heinetal-rec"
DATASET_NAME = "human_ppi_lcqb_s900"
BASE_DIR = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi")
DATA_DIR = BASE_DIR / "data_repository"
RANKMERGING_DIR = BASE_DIR / "Complementarity_rankmerging"
DATASET_DIR = RANKMERGING_DIR / "dataset"
PAIRS_DIR = RANKMERGING_DIR / "potential_pairs"
SCORES_DIR = RANKMERGING_DIR / "scores"
RESULT_DIR = RANKMERGING_DIR / "result/plots"
LEARNING_DIR = RANKMERGING_DIR / "rankmerging"
TRAIN_DATASET = DATASET_DIR / f"train_edges_{DATASET_NAME}.txt"
VAL_DATASET = DATASET_DIR / f"val_edges_{DATASET_NAME}.txt"
TEST_DATASET = DATASET_DIR / f"test_edges_{DATASET_NAME}.txt"
COMBINED_DATASET = PAIRS_DIR / f"combined_{DATASET_NAME}.txt"


