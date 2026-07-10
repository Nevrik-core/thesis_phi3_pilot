import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_NUM_GPU = 0
RUNTIME_PROCESSOR = "CPU"
RANDOM_SEED = 42

UA_QA_SUBSET_SIZE = int(os.getenv("UA_QA_SUBSET_SIZE", "500"))
EN_QA_SUBSET_SIZE = int(os.getenv("EN_QA_SUBSET_SIZE", "500"))
BELEBELE_SUBSET_SIZE = int(os.getenv("BELEBELE_SUBSET_SIZE", "500"))
UALIGN_SUBSET_SIZE = int(os.getenv("UALIGN_SUBSET_SIZE", "500"))

UA_DATASET_NAME = "FIdo-AI/ua-squad"
EN_DATASET_NAME = "squad"

BELEBELE_DATASET_NAME = "facebook/belebele"
BELEBELE_UK_LANG = "ukr_Cyrl"
BELEBELE_EN_LANG = "eng_Latn"

UALIGN_DATASET_NAME = "Stereotypes-in-LLMs/UAlign"
UALIGN_ETHICS_CONFIG = "ETHICS"
UALIGN_SOCIAL_CHEMISTRY_CONFIG = "Social Chemistry 101"

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


def model_cfg(
    *,
    model_name: str,
    display_name: str,
    backend_name: str,
    quantization_name: str,
    source_repo: str,
    artifact_family: str = "GGUF",
    quantization_pipeline: str = "external_gguf",
    imatrix_used: str = "unknown",
    model_role: str = "stage2_quant_curve",
    base_model_family: str = "",
    reasoning_mode: str = "instruct",
    model_size_class: str = "4b",
    architecture_note: str = "",
    prompt_prefix: str = "",
    ollama_think: bool | None = None,
    requires_long_generation_budget: bool = False,
    run_group: str = "stage2_q8_to_q4_500",
) -> dict:
    return {
        "model_name": model_name,
        "display_name": display_name,
        "backend_name": backend_name,
        "quantization_name": quantization_name,
        "source_repo": source_repo,
        "artifact_family": artifact_family,
        "quantization_pipeline": quantization_pipeline,
        "imatrix_used": imatrix_used,
        "model_role": model_role,
        "base_model_family": base_model_family,
        "reasoning_mode": reasoning_mode,
        "model_size_class": model_size_class,
        "architecture_note": architecture_note,
        "prompt_prefix": prompt_prefix,
        "ollama_think": ollama_think,
        "requires_long_generation_budget": requires_long_generation_budget,
        "run_group": run_group,
    }


OLLAMA_GGUF_MODELS = {}


def add_stage2_curve(
    *,
    prefix: str,
    repo_tag_base: str,
    display_base: str,
    source_repo: str,
    base_model_family: str,
    reasoning_mode: str = "instruct",
    prompt_prefix: str = "",
    ollama_think: bool | None = None,
    requires_long_generation_budget: bool = False,
    artifact_family: str = "GGUF",
    quantization_pipeline: str = "external_gguf",
    imatrix_used: str = "unknown",
    quantizations: list[str] | None = None,
) -> list[str]:
    if quantizations is None:
        quantizations = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]

    keys = []

    for quant in quantizations:
        suffix = quant.lower()
        key = f"stage2_{prefix}_{suffix}"

        model_name = f"hf.co/{repo_tag_base}:{quant}"

        OLLAMA_GGUF_MODELS[key] = model_cfg(
            model_name=model_name,
            display_name=f"{display_base} {quant}",
            backend_name=key,
            quantization_name=quant,
            source_repo=source_repo,
            artifact_family=artifact_family,
            quantization_pipeline=quantization_pipeline,
            imatrix_used=imatrix_used,
            base_model_family=base_model_family,
            reasoning_mode=reasoning_mode,
            model_role="stage2_selected_quant_curve",
            architecture_note="stage2_selected_q8_to_q4_curve",
            prompt_prefix=prompt_prefix,
            ollama_think=ollama_think,
            requires_long_generation_budget=requires_long_generation_budget,
        )

        keys.append(key)

    return keys


STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS = []

STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS += add_stage2_curve(
    prefix="phi4_mini_instruct",
    repo_tag_base="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
    display_base="Phi-4-mini-instruct",
    source_repo="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
    base_model_family="phi4",
    quantization_pipeline="llama.cpp_b4792_imatrix",
    imatrix_used="true",
)

STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS += add_stage2_curve(
    prefix="gemma3_4b_it",
    repo_tag_base="bartowski/google_gemma-3-4b-it-GGUF",
    display_base="Gemma-3-4B-it",
    source_repo="bartowski/google_gemma-3-4b-it-GGUF",
    base_model_family="gemma3",
)

STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS += add_stage2_curve(
    prefix="mamaylm_gemma3_4b_it",
    repo_tag_base="INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF",
    display_base="MamayLM-Gemma-3-4B-IT-v1.0",
    source_repo="INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF",
    base_model_family="gemma3_mamay",
    quantizations=["Q8_0", "Q4_K_M"],
)

STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS += add_stage2_curve(
    prefix="qwen3_5_4b",
    repo_tag_base="unsloth/Qwen3.5-4B-GGUF",
    display_base="Qwen3.5-4B /no_think",
    source_repo="unsloth/Qwen3.5-4B-GGUF",
    base_model_family="qwen3.5",
    reasoning_mode="no_think",
    prompt_prefix="/no_think\n",
    ollama_think=False,
)

STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS += add_stage2_curve(
    prefix="minicpm3_4b",
    repo_tag_base="QuantFactory/MiniCPM3-4B-GGUF",
    display_base="MiniCPM3-4B",
    source_repo="QuantFactory/MiniCPM3-4B-GGUF",
    base_model_family="minicpm",
)

STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS += add_stage2_curve(
    prefix="nemotron3_nano_4b_no_think",
    repo_tag_base="unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF",
    display_base="NVIDIA-Nemotron-3-Nano-4B /no_think",
    source_repo="unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF",
    base_model_family="nemotron3",
    reasoning_mode="no_think",
    prompt_prefix="Do not reason. Answer directly and only in the requested format.\n",
    ollama_think=False,
)


STAGE2_UALIGN_EXTRA_REASONING_QUANT_MODEL_KEYS = []

STAGE2_UALIGN_EXTRA_REASONING_QUANT_MODEL_KEYS += add_stage2_curve(
    prefix="nemotron3_nano_4b_think",
    repo_tag_base="nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
    display_base="NVIDIA-Nemotron-3-Nano-4B /think",
    source_repo="nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
    base_model_family="nemotron3",
    reasoning_mode="think",
    prompt_prefix="Reason if needed, but return only the final answer in the requested format.\n",
    ollama_think=True,
    requires_long_generation_budget=True,
)

STAGE2_UALIGN_EXTRA_REASONING_QUANT_MODEL_KEYS += add_stage2_curve(
    prefix="phi4_mini_reasoning",
    repo_tag_base="bartowski/microsoft_Phi-4-mini-reasoning-GGUF",
    display_base="Phi-4-mini-reasoning",
    source_repo="bartowski/microsoft_Phi-4-mini-reasoning-GGUF",
    base_model_family="phi4",
    reasoning_mode="reasoning_only",
    ollama_think=None,
    requires_long_generation_budget=True,
)


STAGE2_SMOKE_QUANT_MODEL_KEYS = (
    STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS
    + STAGE2_UALIGN_EXTRA_REASONING_QUANT_MODEL_KEYS
)

STAGE2_QA_QUANT_MODEL_KEYS = STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS
STAGE2_BELEBELE_QUANT_MODEL_KEYS = STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS
STAGE2_UALIGN_QUANT_MODEL_KEYS = (
    STAGE2_QA_BELEBELE_QUANT_MODEL_KEYS
    + STAGE2_UALIGN_EXTRA_REASONING_QUANT_MODEL_KEYS
)


ACTIVE_OLLAMA_MODEL_KEY = os.getenv(
    "ACTIVE_OLLAMA_MODEL_KEY",
    STAGE2_QA_QUANT_MODEL_KEYS[0],
)


def get_active_model_config() -> dict:
    if ACTIVE_OLLAMA_MODEL_KEY not in OLLAMA_GGUF_MODELS:
        raise KeyError(
            f"Unknown ACTIVE_OLLAMA_MODEL_KEY: {ACTIVE_OLLAMA_MODEL_KEY}. "
            f"Available keys: {list(OLLAMA_GGUF_MODELS.keys())}"
        )

    return OLLAMA_GGUF_MODELS[ACTIVE_OLLAMA_MODEL_KEY]


ACTIVE_MODEL_CONFIG = get_active_model_config()

PRIMARY_MODEL_NAME = ACTIVE_MODEL_CONFIG["model_name"]
PRIMARY_MODEL_DISPLAY_NAME = ACTIVE_MODEL_CONFIG["display_name"]
BACKEND_NAME = ACTIVE_MODEL_CONFIG["backend_name"]
QUANTIZATION_NAME = ACTIVE_MODEL_CONFIG["quantization_name"]
MODEL_SOURCE_REPO = ACTIVE_MODEL_CONFIG["source_repo"]
ARTIFACT_FAMILY = ACTIVE_MODEL_CONFIG["artifact_family"]
QUANTIZATION_PIPELINE = ACTIVE_MODEL_CONFIG["quantization_pipeline"]
IMATRIX_USED = ACTIVE_MODEL_CONFIG["imatrix_used"]
MODEL_ROLE = ACTIVE_MODEL_CONFIG["model_role"]
PROMPT_PREFIX = ACTIVE_MODEL_CONFIG.get("prompt_prefix", "")
OLLAMA_THINK = ACTIVE_MODEL_CONFIG.get("ollama_think")

BASE_MODEL_FAMILY = ACTIVE_MODEL_CONFIG.get("base_model_family", "")
REASONING_MODE = ACTIVE_MODEL_CONFIG.get("reasoning_mode", "")
MODEL_SIZE_CLASS = ACTIVE_MODEL_CONFIG.get("model_size_class", "")
ARCHITECTURE_NOTE = ACTIVE_MODEL_CONFIG.get("architecture_note", "")
RUN_GROUP = ACTIVE_MODEL_CONFIG.get("run_group", "")
REQUIRES_LONG_GENERATION_BUDGET = ACTIVE_MODEL_CONFIG.get(
    "requires_long_generation_budget",
    False,
)

OLLAMA_BACKEND_LABEL = BACKEND_NAME
