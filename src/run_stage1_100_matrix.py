import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    PROJECT_ROOT,
    RESULTS_DIR,
    OLLAMA_GGUF_MODELS,
    STAGE1_CONNECTION_SMOKE_MODEL_KEYS,
)


STAGE1_100_MODEL_KEYS = STAGE1_CONNECTION_SMOKE_MODEL_KEYS


def next_run_version(results_dir: Path, prefix: str = "stage1_q4_100") -> str:
    existing = []

    if not results_dir.exists():
        return f"{prefix}_v1"

    pattern = re.compile(rf"{re.escape(prefix)}_v(\d+)$")

    for path in results_dir.iterdir():
        if not path.is_dir():
            continue

        match = pattern.fullmatch(path.name)

        if match:
            existing.append(int(match.group(1)))

    if not existing:
        return f"{prefix}_v1"

    return f"{prefix}_v{max(existing) + 1}"


def run_live(command: list[str], env: dict[str, str] | None = None, check: bool = True) -> int:
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


def ollama_stop(model_name: str) -> None:
    run_live(["ollama", "stop", model_name], check=False)


def ollama_pull(model_name: str) -> None:
    run_live(["ollama", "pull", model_name], check=True)


def ollama_ps() -> None:
    run_live(["ollama", "ps"], check=False)


def stop_all_stage1_models(model_keys: list[str]) -> None:
    print("\n[STOP ALL STAGE1 MODELS]")

    model_names = {
        OLLAMA_GGUF_MODELS[model_key]["model_name"]
        for model_key in model_keys
    }

    for model_name in sorted(model_names):
        ollama_stop(model_name)

    time.sleep(3)
    ollama_ps()


def save_manifest(
    output_dir: Path,
    experiment_version: str,
    model_keys: list[str],
) -> Path:
    rows: list[dict[str, Any]] = []

    for index, model_key in enumerate(model_keys, start=1):
        cfg = OLLAMA_GGUF_MODELS[model_key]

        rows.append(
            {
                "run_order": index,
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
                "base_model_family": cfg.get("base_model_family", ""),
                "reasoning_mode": cfg.get("reasoning_mode", ""),
                "model_size_class": cfg.get("model_size_class", ""),
                "architecture_note": cfg.get("architecture_note", ""),
                "prompt_prefix": cfg.get("prompt_prefix", ""),
                "ollama_think": cfg.get("ollama_think"),
                "run_group": cfg.get("run_group", ""),
            }
        )

    manifest_df = pd.DataFrame(rows)
    manifest_path = output_dir / f"stage1_100_matrix_manifest_{experiment_version}.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED MANIFEST] {manifest_path}")
    return manifest_path


def run_benchmarks_for_model(
    model_key: str,
    experiment_version: str,
    output_dir: Path,
    model_keys_to_stop: list[str],
    pull: bool,
) -> None:
    if model_key not in OLLAMA_GGUF_MODELS:
        raise KeyError(f"Unknown model key: {model_key}")

    cfg = OLLAMA_GGUF_MODELS[model_key]
    model_name = cfg["model_name"]

    print("\n" + "#" * 100)
    print(f"# STAGE 1 Q4 100-SAMPLE RUN: {model_key}")
    print(f"# Experiment version: {experiment_version}")
    print(f"# Output dir: {output_dir}")
    print(f"# Model: {model_name}")
    print(f"# Display: {cfg['display_name']}")
    print(f"# Quantization: {cfg['quantization_name']}")
    print(f"# Backend: {cfg['backend_name']}")
    print(f"# Family: {cfg.get('base_model_family', '')}")
    print(f"# Reasoning mode: {cfg.get('reasoning_mode', '')}")
    print(f"# Ollama think: {cfg.get('ollama_think')}")
    print(f"# Prompt prefix: {repr(cfg.get('prompt_prefix', ''))}")
    print("#" * 100)

    stop_all_stage1_models(model_keys_to_stop)

    if pull:
        print(f"\n[PULL MODEL] {model_name}")
        ollama_pull(model_name)

    env = os.environ.copy()
    env["ACTIVE_OLLAMA_MODEL_KEY"] = model_key
    env["EXPERIMENT_VERSION"] = experiment_version
    env["EXPERIMENT_OUTPUT_DIR"] = str(output_dir)

    env["UA_QA_SUBSET_SIZE"] = "100"
    env["EN_QA_SUBSET_SIZE"] = "100"
    env["BELEBELE_SUBSET_SIZE"] = "100"
    env["UALIGN_SUBSET_SIZE"] = "100"

    print(f"\n[ACTIVE_OLLAMA_MODEL_KEY={model_key}]")
    print(f"[EXPERIMENT_VERSION={experiment_version}]")
    print(f"[EXPERIMENT_OUTPUT_DIR={output_dir}]")
    print("[SUBSET SIZES] QA uk=100, QA en=100, BELEBELE=100, UAlign per task/lang=100")

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


def write_run_notes(output_dir: Path, experiment_version: str, model_keys: list[str]) -> Path:
    notes_path = output_dir / f"stage1_100_run_notes_{experiment_version}.md"

    lines = [
        f"# Stage 1 Q4 100-sample run: {experiment_version}",
        "",
        "## Scope",
        "",
        "This run evaluates Stage 1 Q4_K_M / Q4 packaged models on 100-sample subsets.",
        "",
        "Benchmarks:",
        "",
        "- QA: 100 Ukrainian + 100 English examples",
        "- BELEBELE: 100 Ukrainian + 100 English examples",
        "- UAlign: 100 examples per task/language pair",
        "",
        "Runtime:",
        "",
        "- Ollama local API",
        "- CPU-only requested with `num_gpu=0`",
        "- temperature=0.0",
        "- num_ctx=2048",
        "",
        "Important:",
        "",
        "- Thinking/reasoning models use larger generation budgets inside the eval scripts.",
        "- `prompt_prefix` and `ollama_think` are taken from `config.py` metadata.",
        "- Outputs include `raw_content`, `thinking`, `clean_text_source`, and `used_thinking_fallback` for later debugging.",
        "",
        "## Models",
        "",
    ]

    for index, model_key in enumerate(model_keys, start=1):
        cfg = OLLAMA_GGUF_MODELS[model_key]
        lines.append(
            f"{index}. `{model_key}` — {cfg['display_name']} "
            f"({cfg.get('reasoning_mode', '')}, think={cfg.get('ollama_think')})"
        )

    notes_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED NOTES] {notes_path}")

    return notes_path


def main() -> None:
    experiment_version = os.getenv(
        "EXPERIMENT_VERSION",
        next_run_version(RESULTS_DIR, prefix="stage1_q4_100"),
    )

    output_dir = RESULTS_DIR / experiment_version
    output_dir.mkdir(parents=True, exist_ok=True)

    model_keys = STAGE1_100_MODEL_KEYS

    no_pull = os.getenv("STAGE1_NO_PULL", "0") == "1"
    pull = not no_pull

    print("\n=== STAGE 1 Q4 100-SAMPLE MATRIX RUNNER ===")
    print(f"Python: {sys.executable}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Experiment version: {experiment_version}")
    print(f"Output dir: {output_dir}")
    print(f"Pull models: {pull}")
    print(f"Models: {len(model_keys)}")

    print("\nModels:")
    for model_key in model_keys:
        cfg = OLLAMA_GGUF_MODELS[model_key]
        print(
            f"- {model_key}: {cfg['display_name']} | "
            f"{cfg.get('base_model_family', '')} | "
            f"{cfg.get('reasoning_mode', '')} | "
            f"think={cfg.get('ollama_think')}"
        )

    print(
        "\nImportant: for true CPU-only mode, run Ollama server with:\n"
        '$env:CUDA_VISIBLE_DEVICES="-1"\n'
        "ollama serve\n"
    )

    save_manifest(
        output_dir=output_dir,
        experiment_version=experiment_version,
        model_keys=model_keys,
    )

    write_run_notes(
        output_dir=output_dir,
        experiment_version=experiment_version,
        model_keys=model_keys,
    )

    for model_key in model_keys:
        run_benchmarks_for_model(
            model_key=model_key,
            experiment_version=experiment_version,
            output_dir=output_dir,
            model_keys_to_stop=model_keys,
            pull=pull,
        )

    print("\n=== STAGE 1 Q4 100-SAMPLE MATRIX FINISHED ===")
    ollama_ps()


if __name__ == "__main__":
    main()