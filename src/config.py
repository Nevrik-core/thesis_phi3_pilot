from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MODEL_NAME = "phi4-mini"
PRIMARY_MODEL_DISPLAY_NAME = "Phi-4-mini-instruct"
BACKEND_NAME = "ollama_local"

# Малий пілот
UA_QA_SUBSET_SIZE = 10
EN_QA_SUBSET_SIZE = 10

GENERATION_CONFIG = {
    "temperature": 0.0,
    "max_new_tokens": 32,
    "do_sample": False,
    "num_ctx": 2048,
}

UA_DATASET_NAME = "FIdo-AI/ua-squad"
EN_DATASET_NAME = "squad"