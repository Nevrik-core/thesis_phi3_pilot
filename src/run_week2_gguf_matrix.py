import os
import subprocess
import sys
import time

from config import (
    PROJECT_ROOT,
    OLLAMA_GGUF_MODELS,
    WEEK2_GGUF_MODEL_KEYS,
)


EXTRA_MODELS_TO_STOP = [
    "phi4-mini",
    "hf.co/professorf/Phi-4-mini-instruct-gguf:BF16",
    "hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q8_0",
    "hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q6_K",
    "hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q5_K_M",
    "hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q4_K_M",
]


def run_live(command: list[str], env: dict | None = None, check: bool = True):
    print("\n" + "=" * 100)
    print("$ " + " ".join(command))
    print("=" * 100)

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
    )

    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {completed.returncode}: {' '.join(command)}"
        )

    return completed.returncode


def ollama_stop(model_name: str):
    run_live(
        ["ollama", "stop", model_name],
        check=False,
    )


def ollama_pull(model_name: str):
    run_live(
        ["ollama", "pull", model_name],
        check=True,
    )


def ollama_ps():
    run_live(
        ["ollama", "ps"],
        check=False,
    )


def stop_all_known_models():
    print("\n[STOP ALL KNOWN OLLAMA MODELS]")

    all_models = set(EXTRA_MODELS_TO_STOP)

    for cfg in OLLAMA_GGUF_MODELS.values():
        all_models.add(cfg["model_name"])

    for model_name in sorted(all_models):
        ollama_stop(model_name)

    time.sleep(3)
    ollama_ps()


def run_benchmarks_for_model(model_key: str):
    if model_key not in OLLAMA_GGUF_MODELS:
        raise KeyError(f"Unknown model key: {model_key}")

    model_cfg = OLLAMA_GGUF_MODELS[model_key]
    model_name = model_cfg["model_name"]

    print("\n" + "#" * 100)
    print(f"# WEEK 2 FULL RUN: {model_key}")
    print(f"# Model: {model_name}")
    print(f"# Quantization: {model_cfg['quantization_name']}")
    print(f"# Backend: {model_cfg['backend_name']}")
    print(f"# Role: {model_cfg['model_role']}")
    print("#" * 100)

    stop_all_known_models()

    print(f"\n[PULL MODEL] {model_name}")
    ollama_pull(model_name)

    env = os.environ.copy()
    env["ACTIVE_OLLAMA_MODEL_KEY"] = model_key

    print(f"\n[ACTIVE_OLLAMA_MODEL_KEY={model_key}]")
    ollama_ps()

    run_live(
        [sys.executable, "src/pilot_qa_eval.py"],
        env=env,
        check=True,
    )

    # Keep the same single model loaded for BELEBELE.
    # This avoids loading another model and keeps memory measurement isolated.
    ollama_ps()

    run_live(
        [sys.executable, "src/pilot_belebele_eval.py"],
        env=env,
        check=True,
    )

    ollama_ps()

    print(f"\n[FINISH MODEL] {model_key}")
    ollama_stop(model_name)
    time.sleep(3)
    ollama_ps()


def main():
    print("\n=== WEEK 2 GGUF MATRIX RUNNER ===")
    print("Models:")
    for key in WEEK2_GGUF_MODEL_KEYS:
        cfg = OLLAMA_GGUF_MODELS[key]
        print(f"- {key}: {cfg['model_name']} ({cfg['quantization_name']})")

    print(
        "\nImportant: for true CPU-only mode, run Ollama server with:\n"
        '$env:CUDA_VISIBLE_DEVICES="-1"\n'
        "ollama serve\n"
    )

    for model_key in WEEK2_GGUF_MODEL_KEYS:
        run_benchmarks_for_model(model_key)

    print("\n=== WEEK 2 GGUF MATRIX FINISHED ===")
    ollama_ps()


if __name__ == "__main__":
    main()