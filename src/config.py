from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
STORE_FILE = RAW_DATA_DIR / "store.csv"

TRAIN_PROCESSED_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED_FILE = PROCESSED_DATA_DIR / "test_processed.csv"

MODEL_FILE = MODEL_DIR / "retail_demand_model.pkl"