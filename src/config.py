from pathlib import Path

# ===== Project paths =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===== Pilot experiment settings =====
PRIMARY_MODEL_NAME = "phi4_mini"
PRIMARY_MODEL_DISPLAY_NAME = "Phi-4-mini-instruct"

# Поки що це просто мітка.
# Конкретний локальний backend підставимо після smoke test.
BACKEND_NAME = "local_backend_tbd"

# Для першого пілота робимо маленькі піднабори
UA_QA_SUBSET_SIZE = 20
EN_QA_SUBSET_SIZE = 20

# Детермінований режим для чесного порівняння
GENERATION_CONFIG = {
    "temperature": 0.0,
    "max_new_tokens": 32,
    "do_sample": False,
}

# ===== Dataset names =====
UA_DATASET_NAME = "FIdo-AI/ua-squad"
EN_DATASET_NAME = "squad"