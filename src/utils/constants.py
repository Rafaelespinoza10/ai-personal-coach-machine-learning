from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
FINAL_DATA_DIR = BASE_DIR / 'datasets' / 'final'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_COL = 'stress_level_norm'

EXCLUDE_COLS = ['dataset_source', TARGET_COL]