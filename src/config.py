from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MODEL_NAME = "phi4-mini"
PRIMARY_MODEL_DISPLAY_NAME = "Phi-4-mini-instruct"
BACKEND_NAME = "ollama_cpu_q4_k_m"
OLLAMA_NUM_GPU = 0
OLLAMA_BACKEND_LABEL = "ollama_cpu_q4_k_m"


# QA pilot / benchmark sizes
UA_QA_SUBSET_SIZE = 100
EN_QA_SUBSET_SIZE = 100

# BELEBELE
BELEBELE_DATASET_NAME = "facebook/belebele"
BELEBELE_UK_LANG = "ukr_Cyrl"
BELEBELE_EN_LANG = "eng_Latn"
BELEBELE_SUBSET_SIZE = 100 

GENERATION_CONFIG = {
    "temperature": 0.0,
    "max_new_tokens": 32,
    "do_sample": False,
    "num_ctx": 2048,
}

MC_GENERATION_CONFIG = {
    "temperature": 0.0,
    "max_new_tokens": 4,
    "do_sample": False,
    "num_ctx": 2048,
}

UA_DATASET_NAME = "FIdo-AI/ua-squad"
EN_DATASET_NAME = "squad"

RANDOM_SEED = 42