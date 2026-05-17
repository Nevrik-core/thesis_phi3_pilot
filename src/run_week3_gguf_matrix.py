# src/run_week3_gguf_matrix.py
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from config import (
    PROJECT_ROOT,
    RESULTS_DIR,
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


def next_run_version(results_dir: Path) -> str:
    existing = []

    for path in results_dir.iterdir():
        if not path.is_dir():
            continue

        match = re.fullmatch(r"v(\d+)", path.name)

        if match:
            existing.append(int(match.group(1)))

    if not existing:
        return "v1"

    return f"v{max(existing) + 1}"


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


def run_benchmarks_for_model(
    model_key: str,
    experiment_version: str,
    output_dir: Path,
):
    if model_key not in OLLAMA_GGUF_MODELS:
        raise KeyError(f"Unknown model key: {model_key}")

    model_cfg = OLLAMA_GGUF_MODELS[model_key]
    model_name = model_cfg["model_name"]

    print("\n" + "#" * 100)
    print(f"# WEEK 3 FULL RUN: {model_key}")
    print(f"# Experiment version: {experiment_version}")
    print(f"# Output dir: {output_dir}")
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
    env["EXPERIMENT_VERSION"] = experiment_version
    env["EXPERIMENT_OUTPUT_DIR"] = str(output_dir)

    print(f"\n[ACTIVE_OLLAMA_MODEL_KEY={model_key}]")
    print(f"[EXPERIMENT_VERSION={experiment_version}]")
    print(f"[EXPERIMENT_OUTPUT_DIR={output_dir}]")
    ollama_ps()

    run_live(
        [sys.executable, "src/pilot_qa_eval.py"],
        env=env,
        check=True,
    )

    ollama_ps()

    run_live(
        [sys.executable, "src/pilot_belebele_eval.py"],
        env=env,
        check=True,
    )

    ollama_ps()

    run_live(
        [sys.executable, "src/pilot_ualign_eval.py"],
        env=env,
        check=True,
    )

    ollama_ps()

    print(f"\n[FINISH MODEL] {model_key}")
    ollama_stop(model_name)
    time.sleep(3)
    ollama_ps()


def save_manifest(
    output_dir: Path,
    experiment_version: str,
    model_keys: list[str],
):
    rows = []

    for model_key in model_keys:
        cfg = OLLAMA_GGUF_MODELS[model_key]

        rows.append(
            {
                "experiment_version": experiment_version,
                "model_key": model_key,
                "model_name": cfg["model_name"],
                "display_name": cfg["display_name"],
                "backend_name": cfg["backend_name"],
                "quantization_name": cfg["quantization_name"],
                "source_repo": cfg["source_repo"],
                "artifact_family": cfg["artifact_family"],
                "quantization_pipeline": cfg["quantization_pipeline"],
                "imatrix_used": cfg["imatrix_used"],
                "model_role": cfg["model_role"],
            }
        )

    manifest_df = pd.DataFrame(rows)
    manifest_path = output_dir / f"week3_matrix_manifest_{experiment_version}.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED MANIFEST] {manifest_path}")


def main():
    experiment_version = next_run_version(RESULTS_DIR)
    output_dir = RESULTS_DIR / experiment_version
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== WEEK 3 GGUF MATRIX RUNNER ===")
    print(f"Experiment version: {experiment_version}")
    print(f"Output dir: {output_dir}")

    print("\nModels:")
    for key in WEEK2_GGUF_MODEL_KEYS:
        cfg = OLLAMA_GGUF_MODELS[key]
        print(f"- {key}: {cfg['model_name']} ({cfg['quantization_name']})")

    print(
        "\nImportant: for true CPU-only mode, run Ollama server with:\n"
        '$env:CUDA_VISIBLE_DEVICES="-1"\n'
        "ollama serve\n"
    )

    save_manifest(
        output_dir=output_dir,
        experiment_version=experiment_version,
        model_keys=WEEK2_GGUF_MODEL_KEYS,
    )

    for model_key in WEEK2_GGUF_MODEL_KEYS:
        run_benchmarks_for_model(
            model_key=model_key,
            experiment_version=experiment_version,
            output_dir=output_dir,
        )

    print("\n=== WEEK 3 GGUF MATRIX FINISHED ===")
    ollama_ps()

    final_env = os.environ.copy()
    final_env["EXPERIMENT_VERSION"] = experiment_version
    final_env["EXPERIMENT_OUTPUT_DIR"] = str(output_dir)

    print("\n[BUILD WEEK 3 FIGURES]")
    run_live(
        [sys.executable, "src/build_week3_figures.py"],
        env=final_env,
        check=True,
    )

    print("\n[BUILD QA ERROR ANALYSIS]")
    run_live(
        [sys.executable, "src/qa_error_analysis.py"],
        env=final_env,
        check=True,
    )


if __name__ == "__main__":
    main()