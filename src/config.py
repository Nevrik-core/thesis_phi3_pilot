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

# CPU-only режим.

OLLAMA_NUM_GPU = 0
RUNTIME_PROCESSOR = "CPU"

RANDOM_SEED = 42


# ============================================================
# Dataset sizes
# ============================================================

UA_QA_SUBSET_SIZE = 100
EN_QA_SUBSET_SIZE = 100

BELEBELE_SUBSET_SIZE = 100


# ============================================================
# Datasets
# ============================================================

UA_DATASET_NAME = "FIdo-AI/ua-squad"
EN_DATASET_NAME = "squad"

BELEBELE_DATASET_NAME = "facebook/belebele"
BELEBELE_UK_LANG = "ukr_Cyrl"
BELEBELE_EN_LANG = "eng_Latn"


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
# GGUF model registry
# ============================================================

OLLAMA_GGUF_MODELS = {
    # ------------------------------------------------------------
    # Existing Ollama library model.
    # Kept only for historical/reference runs.
    # This was the first Q4 model used before we moved to a cleaner
    # GGUF same-runtime matrix.
    # ------------------------------------------------------------
    "ollama_library_q4_k_m": {
        "model_name": "phi4-mini",
        "display_name": "Phi-4-mini-instruct",
        "backend_name": "ollama_cpu_q4_k_m",
        "quantization_name": "Q4_K_M",
        "source_repo": "ollama_library/phi4-mini",
        "artifact_family": "ollama_library",
        "quantization_pipeline": "unknown_ollama_packaged_gguf",
        "imatrix_used": "unknown",
        "model_role": "historical_reference",
    },

    # ------------------------------------------------------------
    # External high-precision GGUF reference.
    #
    # Professorf provides BF16 GGUF:
    # hf.co/professorf/Phi-4-mini-instruct-gguf:BF16
    #
    # This is not part of the Bartowski quantization curve,
    # but it is useful as a high-precision GGUF reference.
    # ------------------------------------------------------------
    "professorf_bf16": {
        "model_name": "hf.co/professorf/Phi-4-mini-instruct-gguf:BF16",
        "display_name": "Phi-4-mini-instruct GGUF BF16",
        "backend_name": "ollama_cpu_gguf_professorf_bf16",
        "quantization_name": "BF16",
        "source_repo": "professorf/Phi-4-mini-instruct-gguf",
        "artifact_family": "GGUF",
        "quantization_pipeline": "external_high_precision_gguf_reference",
        "imatrix_used": "not_applicable_bf16",
        "model_role": "external_high_precision_reference",
    },

    # ------------------------------------------------------------
    # Main clean GGUF curve: Bartowski.
    #
    # These models are used as the main same-runtime quantization
    # comparison curve:
    #
    # Q8_0 -> Q6_K -> Q5_K_M -> Q4_K_M
    #
    # Same model family, same runtime target, same source repo,
    # same documented quantization pipeline.
    # ------------------------------------------------------------
    "bartowski_q8_0": {
        "model_name": "hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q8_0",
        "display_name": "Phi-4-mini-instruct GGUF Q8_0",
        "backend_name": "ollama_cpu_gguf_bartowski_q8_0",
        "quantization_name": "Q8_0",
        "source_repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "artifact_family": "GGUF",
        "quantization_pipeline": "llama.cpp_b4792_imatrix",
        "imatrix_used": "true",
        "model_role": "main_same_runtime_curve",
    },

    "bartowski_q6_k": {
        "model_name": "hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q6_K",
        "display_name": "Phi-4-mini-instruct GGUF Q6_K",
        "backend_name": "ollama_cpu_gguf_bartowski_q6_k",
        "quantization_name": "Q6_K",
        "source_repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "artifact_family": "GGUF",
        "quantization_pipeline": "llama.cpp_b4792_imatrix",
        "imatrix_used": "true",
        "model_role": "main_same_runtime_curve",
    },

    "bartowski_q5_k_m": {
        "model_name": "hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q5_K_M",
        "display_name": "Phi-4-mini-instruct GGUF Q5_K_M",
        "backend_name": "ollama_cpu_gguf_bartowski_q5_k_m",
        "quantization_name": "Q5_K_M",
        "source_repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "artifact_family": "GGUF",
        "quantization_pipeline": "llama.cpp_b4792_imatrix",
        "imatrix_used": "true",
        "model_role": "main_same_runtime_curve",
    },

    "bartowski_q4_k_m": {
        "model_name": "hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q4_K_M",
        "display_name": "Phi-4-mini-instruct GGUF Q4_K_M",
        "backend_name": "ollama_cpu_gguf_bartowski_q4_k_m",
        "quantization_name": "Q4_K_M",
        "source_repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "artifact_family": "GGUF",
        "quantization_pipeline": "llama.cpp_b4792_imatrix",
        "imatrix_used": "true",
        "model_role": "main_same_runtime_curve",
    },

    # ------------------------------------------------------------
    # Optional extreme compression.
    #
    # In direct Ollama HF tag smoke test, Q3_K_M returned 404.
    # Keep this in config for later manual Modelfile usage, but do
    # not include it in WEEK2_GGUF_MODEL_KEYS yet.
    # ------------------------------------------------------------
    "bartowski_q3_k_m": {
        "model_name": "hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q3_K_M",
        "display_name": "Phi-4-mini-instruct GGUF Q3_K_M",
        "backend_name": "ollama_cpu_gguf_bartowski_q3_k_m",
        "quantization_name": "Q3_K_M",
        "source_repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "artifact_family": "GGUF",
        "quantization_pipeline": "llama.cpp_b4792_imatrix",
        "imatrix_used": "true",
        "model_role": "optional_extreme_compression",
    },

    # ------------------------------------------------------------
    # Cross-source sanity check.
    #
    # Optional later: compare Professorf Q4_K_M vs Bartowski Q4_K_M
    # to see whether source/conversion pipeline affects the result.
    # Not part of the main Week 2 matrix.
    # ------------------------------------------------------------
    "professorf_q4_k_m": {
        "model_name": "hf.co/professorf/Phi-4-mini-instruct-gguf:Q4_K_M",
        "display_name": "Phi-4-mini-instruct GGUF Q4_K_M Professorf",
        "backend_name": "ollama_cpu_gguf_professorf_q4_k_m",
        "quantization_name": "Q4_K_M",
        "source_repo": "professorf/Phi-4-mini-instruct-gguf",
        "artifact_family": "GGUF",
        "quantization_pipeline": "external_reference_unknown_or_less_documented",
        "imatrix_used": "unknown",
        "model_role": "cross_source_sanity_check",
    },
}


# ============================================================
# Active model selection
# ============================================================

# Default active model if no environment variable is provided.
#
# You can override it without editing this file:
#
# PowerShell:
# $env:ACTIVE_OLLAMA_MODEL_KEY="professorf_bf16"
# python src/pilot_qa_eval.py
#
# Or runner script can pass ACTIVE_OLLAMA_MODEL_KEY automatically.
ACTIVE_OLLAMA_MODEL_KEY = os.getenv(
    "ACTIVE_OLLAMA_MODEL_KEY",
    "bartowski_q4_k_m",
)


# Main Week 2 full-run matrix.
#
# This is the set we want for the report:
# BF16 reference + Bartowski Q8/Q6/Q5/Q4 same-runtime curve.
WEEK2_GGUF_MODEL_KEYS = [
    "professorf_bf16",
    "bartowski_q8_0",
    "bartowski_q6_k",
    "bartowski_q5_k_m",
    "bartowski_q4_k_m",
]


# Smoke-test model list.
#
# This can include optional/problematic models because smoke is allowed
# to discover what works and what does not.
SMOKE_GGUF_MODEL_KEYS = [
    "professorf_bf16",
    "bartowski_q8_0",
    "bartowski_q6_k",
    "bartowski_q5_k_m",
    "bartowski_q4_k_m",
    "bartowski_q3_k_m",
    "professorf_q4_k_m",
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

OLLAMA_BACKEND_LABEL = BACKEND_NAME