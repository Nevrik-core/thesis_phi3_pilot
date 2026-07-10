import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Runtime settings
# ============================================================

OLLAMA_NUM_GPU = 0
RUNTIME_PROCESSOR = "CPU"
RANDOM_SEED = 42


# ============================================================
# Dataset sizes
# ============================================================

UA_QA_SUBSET_SIZE = int(os.getenv("UA_QA_SUBSET_SIZE", "1000"))
EN_QA_SUBSET_SIZE = int(os.getenv("EN_QA_SUBSET_SIZE", "1000"))

BELEBELE_SUBSET_SIZE = int(os.getenv("BELEBELE_SUBSET_SIZE", "1000"))
UALIGN_SUBSET_SIZE = int(os.getenv("UALIGN_SUBSET_SIZE", "1000"))


# ============================================================
# Datasets
# ============================================================

UA_DATASET_NAME = "FIdo-AI/ua-squad"
EN_DATASET_NAME = "squad"

BELEBELE_DATASET_NAME = "facebook/belebele"
BELEBELE_UK_LANG = "ukr_Cyrl"
BELEBELE_EN_LANG = "eng_Latn"

UALIGN_DATASET_NAME = "Stereotypes-in-LLMs/UAlign"
UALIGN_ETHICS_CONFIG = "ETHICS"
UALIGN_SOCIAL_CHEMISTRY_CONFIG = "Social Chemistry 101"


# ============================================================
# Generation configs
# ============================================================

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


# ============================================================
# GGUF model registry helpers
# ============================================================

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
    model_role: str = "stage1_4b_smoke",
    base_model_family: str = "",
    reasoning_mode: str = "instruct",
    model_size_class: str = "4b",
    architecture_note: str = "",
    prompt_prefix: str = "",
    ollama_think: bool | None = None,
    run_group: str = "stage1_4b_smoke",
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
        "run_group": run_group,
    }


# ============================================================
# GGUF model registry
# ============================================================

OLLAMA_GGUF_MODELS = {
    # ------------------------------------------------------------
    # Historical / thesis Phi-4-mini-instruct entries
    # ------------------------------------------------------------
    "ollama_library_q4_k_m": model_cfg(
        model_name="phi4-mini",
        display_name="Phi-4-mini-instruct",
        backend_name="ollama_cpu_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="ollama_library/phi4-mini",
        artifact_family="ollama_library",
        quantization_pipeline="unknown_ollama_packaged_gguf",
        imatrix_used="unknown",
        model_role="historical_reference",
        base_model_family="phi4",
        reasoning_mode="instruct",
        architecture_note="historical_ollama_library_model",
        run_group="thesis_v4",
    ),

    "professorf_bf16": model_cfg(
        model_name="hf.co/professorf/Phi-4-mini-instruct-gguf:BF16",
        display_name="Phi-4-mini-instruct GGUF BF16",
        backend_name="ollama_cpu_gguf_professorf_bf16",
        quantization_name="BF16",
        source_repo="professorf/Phi-4-mini-instruct-gguf",
        artifact_family="GGUF",
        quantization_pipeline="external_high_precision_gguf_reference",
        imatrix_used="not_applicable_bf16",
        model_role="external_high_precision_reference",
        base_model_family="phi4",
        reasoning_mode="instruct",
        architecture_note="external_bf16_reference",
        run_group="thesis_v4",
    ),

    "bartowski_q8_0": model_cfg(
        model_name="hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q8_0",
        display_name="Phi-4-mini-instruct GGUF Q8_0",
        backend_name="ollama_cpu_gguf_bartowski_q8_0",
        quantization_name="Q8_0",
        source_repo="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        artifact_family="GGUF",
        quantization_pipeline="llama.cpp_b4792_imatrix",
        imatrix_used="true",
        model_role="main_same_runtime_curve",
        base_model_family="phi4",
        reasoning_mode="instruct",
        architecture_note="thesis_main_quantization_curve",
        run_group="thesis_v4",
    ),

    "bartowski_q6_k": model_cfg(
        model_name="hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q6_K",
        display_name="Phi-4-mini-instruct GGUF Q6_K",
        backend_name="ollama_cpu_gguf_bartowski_q6_k",
        quantization_name="Q6_K",
        source_repo="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        artifact_family="GGUF",
        quantization_pipeline="llama.cpp_b4792_imatrix",
        imatrix_used="true",
        model_role="main_same_runtime_curve",
        base_model_family="phi4",
        reasoning_mode="instruct",
        architecture_note="thesis_main_quantization_curve",
        run_group="thesis_v4",
    ),

    "bartowski_q5_k_m": model_cfg(
        model_name="hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q5_K_M",
        display_name="Phi-4-mini-instruct GGUF Q5_K_M",
        backend_name="ollama_cpu_gguf_bartowski_q5_k_m",
        quantization_name="Q5_K_M",
        source_repo="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        artifact_family="GGUF",
        quantization_pipeline="llama.cpp_b4792_imatrix",
        imatrix_used="true",
        model_role="main_same_runtime_curve",
        base_model_family="phi4",
        reasoning_mode="instruct",
        architecture_note="thesis_main_quantization_curve",
        run_group="thesis_v4",
    ),

    "bartowski_q4_k_m": model_cfg(
        model_name="hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q4_K_M",
        display_name="Phi-4-mini-instruct GGUF Q4_K_M",
        backend_name="ollama_cpu_gguf_bartowski_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        artifact_family="GGUF",
        quantization_pipeline="llama.cpp_b4792_imatrix",
        imatrix_used="true",
        model_role="main_same_runtime_curve",
        base_model_family="phi4",
        reasoning_mode="instruct",
        architecture_note="thesis_main_quantization_curve",
        run_group="thesis_v4",
    ),

    "professorf_q4_k_m": model_cfg(
        model_name="hf.co/professorf/Phi-4-mini-instruct-gguf:Q4_K_M",
        display_name="Phi-4-mini-instruct GGUF Q4_K_M Professorf",
        backend_name="ollama_cpu_gguf_professorf_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="professorf/Phi-4-mini-instruct-gguf",
        artifact_family="GGUF",
        quantization_pipeline="external_reference_unknown_or_less_documented",
        imatrix_used="unknown",
        model_role="cross_source_sanity_check",
        base_model_family="phi4",
        reasoning_mode="instruct",
        architecture_note="cross_source_sanity_check",
        run_group="thesis_v4",
    ),

    # ------------------------------------------------------------
    # Stage 1: 4B / near-4B Q4 smoke candidates
    # ------------------------------------------------------------
    "stage1_phi4_mini_instruct_q4_k_m": model_cfg(
        model_name="hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q4_K_M",
        display_name="Phi-4-mini-instruct Q4_K_M",
        backend_name="stage1_phi4_mini_instruct_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        quantization_pipeline="llama.cpp_b4792_imatrix",
        imatrix_used="true",
        base_model_family="phi4",
        reasoning_mode="instruct",
        architecture_note="dense_4b_instruct_baseline",
    ),

    "stage1_phi4_mini_reasoning_q4_k_m": model_cfg(
        model_name="hf.co/bartowski/microsoft_Phi-4-mini-reasoning-GGUF:Q4_K_M",
        display_name="Phi-4-mini-reasoning Q4_K_M",
        backend_name="stage1_phi4_mini_reasoning_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="bartowski/microsoft_Phi-4-mini-reasoning-GGUF",
        base_model_family="phi4",
        reasoning_mode="reasoning_only",
        architecture_note="dense_4b_reasoning",
        ollama_think=None,
    ),

    "stage1_qwen3_4b_instruct_2507_q4_k_m": model_cfg(
        model_name="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M",
        display_name="Qwen3-4B-Instruct-2507 Q4_K_M",
        backend_name="stage1_qwen3_4b_instruct_2507_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
        base_model_family="qwen3",
        reasoning_mode="instruct",
        architecture_note="dense_4b_instruct",
        ollama_think=False,
    ),

    "stage1_qwen3_4b_thinking_2507_q4_k_m": model_cfg(
        model_name="hf.co/bartowski/Qwen_Qwen3-4B-Thinking-2507-GGUF:Q4_K_M",
        display_name="Qwen3-4B-Thinking-2507 Q4_K_M",
        backend_name="stage1_qwen3_4b_thinking_2507_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="bartowski/Qwen_Qwen3-4B-Thinking-2507-GGUF",
        base_model_family="qwen3",
        reasoning_mode="reasoning_only",
        architecture_note="dense_4b_thinking",
        ollama_think=True,
    ),

    "stage1_qwen3_5_4b_q4_k_m": model_cfg(
        model_name="qwen3.5:4b-q4_K_M",
        display_name="Qwen3.5-4B Q4_K_M",
        backend_name="stage1_qwen3_5_4b_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="ollama_library/qwen3.5:4b-q4_K_M",
        artifact_family="ollama_library",
        quantization_pipeline="ollama_packaged_gguf_unknown",
        imatrix_used="unknown",
        base_model_family="qwen3.5",
        reasoning_mode="default_or_hybrid",
        architecture_note="ollama_library_qwen3_5_4b_q4_k_m",
        ollama_think=False,
    ),

    "stage1_gemma3_4b_it_q4_k_m": model_cfg(
        model_name="hf.co/bartowski/google_gemma-3-4b-it-GGUF:Q4_K_M",
        display_name="Gemma-3-4B-it Q4_K_M",
        backend_name="stage1_gemma3_4b_it_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="bartowski/google_gemma-3-4b-it-GGUF",
        base_model_family="gemma3",
        reasoning_mode="instruct",
        architecture_note="dense_4b_instruct",
    ),

    "stage1_mamaylm_gemma3_4b_it_q4_k_m": model_cfg(
        model_name="hf.co/INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF:Q4_K_M",
        display_name="MamayLM-Gemma-3-4B-IT-v1.0 Q4_K_M",
        backend_name="stage1_mamaylm_gemma3_4b_it_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF",
        base_model_family="gemma3_mamay",
        reasoning_mode="instruct",
        architecture_note="ukrainian_adapted_gemma3_4b",
    ),

    "stage1_hunyuan_4b_instruct_no_think_q4_k_m": model_cfg(
        model_name="hf.co/bartowski/tencent_Hunyuan-4B-Instruct-GGUF:Q4_K_M",
        display_name="Hunyuan-4B-Instruct Q4_K_M /no_think",
        backend_name="stage1_hunyuan_4b_instruct_no_think_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="bartowski/tencent_Hunyuan-4B-Instruct-GGUF",
        base_model_family="hunyuan",
        reasoning_mode="no_think",
        architecture_note="dense_4b_think_controlled",
        prompt_prefix="/no_think\n",
        ollama_think=False,
    ),

    "stage1_hunyuan_4b_instruct_think_q4_k_m": model_cfg(
        model_name="hf.co/bartowski/tencent_Hunyuan-4B-Instruct-GGUF:Q4_K_M",
        display_name="Hunyuan-4B-Instruct Q4_K_M /think",
        backend_name="stage1_hunyuan_4b_instruct_think_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="bartowski/tencent_Hunyuan-4B-Instruct-GGUF",
        base_model_family="hunyuan",
        reasoning_mode="think",
        architecture_note="dense_4b_think_controlled",
        prompt_prefix="/think\n",
        ollama_think=True,
    ),

    "stage1_minicpm3_4b_q4_k_m": model_cfg(
        model_name="hf.co/QuantFactory/MiniCPM3-4B-GGUF:Q4_K_M",
        display_name="MiniCPM3-4B Q4_K_M",
        backend_name="stage1_minicpm3_4b_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="QuantFactory/MiniCPM3-4B-GGUF",
        base_model_family="minicpm",
        reasoning_mode="instruct",
        architecture_note="dense_4b_chinese_model",
    ),

    "stage1_llm_jp_3_3_7b_instruct_q4_k_m": model_cfg(
        model_name="hf.co/mmnga/llm-jp-3-3.7b-instruct3-gguf:Q4_K_M",
        display_name="llm-jp-3-3.7b-instruct Q4_K_M",
        backend_name="stage1_llm_jp_3_3_7b_instruct_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="mmnga/llm-jp-3-3.7b-instruct3-gguf",
        base_model_family="llm_jp",
        reasoning_mode="instruct",
        model_size_class="near_4b",
        architecture_note="japanese_language_specialized_near_4b",
    ),

    "stage1_nemotron3_nano_4b_no_think_q4_k_m": model_cfg(
        model_name="hf.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M",
        display_name="NVIDIA-Nemotron-3-Nano-4B Q4_K_M /no_think",
        backend_name="stage1_nemotron3_nano_4b_no_think_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        base_model_family="nemotron3",
        reasoning_mode="no_think",
        architecture_note="hybrid_mamba_transformer_4b",
        prompt_prefix="Do not reason. Answer directly and only in the requested format.\n",
        ollama_think=False,
    ),

    "stage1_nemotron3_nano_4b_think_q4_k_m": model_cfg(
        model_name="hf.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M",
        display_name="NVIDIA-Nemotron-3-Nano-4B Q4_K_M /think",
        backend_name="stage1_nemotron3_nano_4b_think_q4_k_m",
        quantization_name="Q4_K_M",
        source_repo="nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        base_model_family="nemotron3",
        reasoning_mode="think",
        architecture_note="hybrid_mamba_transformer_4b",
        prompt_prefix="Reason if needed, but return only the final answer in the requested format.\n",
        ollama_think=True,
    ),
}


# ============================================================
# Active model selections
# ============================================================

ACTIVE_OLLAMA_MODEL_KEY = os.getenv(
    "ACTIVE_OLLAMA_MODEL_KEY",
    "bartowski_q4_k_m",
)

WEEK2_GGUF_MODEL_KEYS = [
    "professorf_bf16",
    "bartowski_q8_0",
    "bartowski_q6_k",
    "bartowski_q5_k_m",
    "bartowski_q4_k_m",
]

SMOKE_GGUF_MODEL_KEYS = [
    "professorf_bf16",
    "bartowski_q8_0",
    "bartowski_q6_k",
    "bartowski_q5_k_m",
    "bartowski_q4_k_m",
]

STAGE1_CONNECTION_SMOKE_MODEL_KEYS = [
    "stage1_phi4_mini_instruct_q4_k_m",
    # "stage1_phi4_mini_reasoning_q4_k_m",
    "stage1_qwen3_4b_instruct_2507_q4_k_m",
    "stage1_qwen3_4b_thinking_2507_q4_k_m",
    "stage1_qwen3_5_4b_q4_k_m",
    "stage1_gemma3_4b_it_q4_k_m",
    "stage1_mamaylm_gemma3_4b_it_q4_k_m",
    "stage1_hunyuan_4b_instruct_no_think_q4_k_m",
    "stage1_hunyuan_4b_instruct_think_q4_k_m",
    "stage1_minicpm3_4b_q4_k_m",
    "stage1_llm_jp_3_3_7b_instruct_q4_k_m",
    "stage1_nemotron3_nano_4b_no_think_q4_k_m",
    "stage1_nemotron3_nano_4b_think_q4_k_m",
]

STAGE1_PROBLEM_SMOKE_MODEL_KEYS = [
    "stage1_qwen3_4b_instruct_2507_q4_k_m",
    "stage1_qwen3_4b_thinking_2507_q4_k_m",
    "stage1_qwen3_5_4b_q4_k_m",
    "stage1_hunyuan_4b_instruct_no_think_q4_k_m",
    "stage1_hunyuan_4b_instruct_think_q4_k_m",
    "stage1_nemotron3_nano_4b_no_think_q4_k_m",
    "stage1_nemotron3_nano_4b_think_q4_k_m",
]

STAGE2_SELECTED_MODEL_KEYS = [
    "stage1_phi4_mini_instruct_q4_k_m",
    "stage1_qwen3_5_4b_q4_k_m",
    "stage1_gemma3_4b_it_q4_k_m",
    "stage1_mamaylm_gemma3_4b_it_q4_k_m",
    "stage1_minicpm3_4b_q4_k_m",
    "stage1_nemotron3_nano_4b_no_think_q4_k_m",
]

STAGE2_UALIGN_REASONING_MODEL_KEYS = [
    "stage1_nemotron3_nano_4b_think_q4_k_m",
]


STAGE2_SELECTED_QUANT_MODEL_KEYS = [
    # Phi-4-mini-instruct
    "stage2_phi4_mini_instruct_q8_0",
    "stage2_phi4_mini_instruct_q6_k",
    "stage2_phi4_mini_instruct_q5_k_m",
    "stage2_phi4_mini_instruct_q4_k_m",

    # Gemma-3-4B-it
    "stage2_gemma3_4b_it_q8_0",
    "stage2_gemma3_4b_it_q6_k",
    "stage2_gemma3_4b_it_q5_k_m",
    "stage2_gemma3_4b_it_q4_k_m",

    # MamayLM
    "stage2_mamaylm_gemma3_4b_it_q8_0",
    "stage2_mamaylm_gemma3_4b_it_q6_k",
    "stage2_mamaylm_gemma3_4b_it_q5_k_m",
    "stage2_mamaylm_gemma3_4b_it_q4_k_m",

    # MiniCPM3
    "stage2_minicpm3_4b_q8_0",
    "stage2_minicpm3_4b_q6_k",
    "stage2_minicpm3_4b_q5_k_m",
    "stage2_minicpm3_4b_q4_k_m",

    # Nemotron /no_think
    "stage2_nemotron3_nano_4b_no_think_q8_0",
    "stage2_nemotron3_nano_4b_no_think_q6_k",
    "stage2_nemotron3_nano_4b_no_think_q5_k_m",
    "stage2_nemotron3_nano_4b_no_think_q4_k_m",

    # Qwen3.5 Ollama library: confirmed practical pair
    "stage2_qwen3_5_4b_q8_0",
    "stage2_qwen3_5_4b_q4_k_m",
]

STAGE2_UALIGN_REASONING_QUANT_MODEL_KEYS = [
    "stage2_nemotron3_nano_4b_think_q8_0",
    "stage2_nemotron3_nano_4b_think_q6_k",
    "stage2_nemotron3_nano_4b_think_q5_k_m",
    "stage2_nemotron3_nano_4b_think_q4_k_m",
]


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

OLLAMA_BACKEND_LABEL = BACKEND_NAME