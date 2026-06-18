import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from config import (
    RESULTS_DIR,
    OLLAMA_NUM_GPU,
    SMOKE_GGUF_MODEL_KEYS,
    OLLAMA_GGUF_MODELS,
    GENERATION_CONFIG,
    MC_GENERATION_CONFIG,
)
from ollama_runner import call_ollama_chat


def next_experiment_version(results_dir: Path, prefix: str) -> int:
    existing_versions = []

    for path in results_dir.glob(f"{prefix}_v*.csv"):
        match = re.search(r"_v(\d+)\.csv$", path.name)

        if match:
            existing_versions.append(int(match.group(1)))

    if not existing_versions:
        return 1

    return max(existing_versions) + 1


def run_command(command: list[str], timeout: int = 120) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

        return "\n".join(
            [
                f"$ {' '.join(command)}",
                f"returncode: {completed.returncode}",
                completed.stdout or "",
                completed.stderr or "",
            ]
        )

    except Exception as exc:
        return f"$ {' '.join(command)}\nERROR: {repr(exc)}"


def stop_model(model_name: str):
    print(f"\n[STOP] ollama stop {model_name}")
    output = run_command(["ollama", "stop", model_name], timeout=120)
    print(output)

    # Small pause so the process can release memory before the next model.
    time.sleep(3)


def print_ollama_ps(label: str) -> str:
    print(f"\n[OLLAMA PS] {label}")
    output = run_command(["ollama", "ps"], timeout=120)
    print(output)
    return output


def qa_smoke_prompt_uk() -> str:
    return """Прочитай контекст і дай коротку точну відповідь на запитання.
Відповідай лише самою відповіддю, без пояснень.

Контекст:
Нормандія розташована у Франції.

Запитання:
У якій країні розташована Нормандія?

Відповідь:"""


def belebele_smoke_prompt_uk() -> str:
    return """Прочитай текст і вибери правильну відповідь.

Ти маєш відповісти рівно одним символом: латинською літерою A, B, C або D.
Не використовуй кириличні літери: А, Б, В, Г, Д.
Не пояснюй. Не пиши слів. Поверни тільки одну латинську літеру.

Текст:
Нормандія — історичний регіон на півночі Франції.

Питання:
У якій країні розташована Нормандія?

A. Німеччина
B. Франція
C. Іспанія
D. Італія

Відповідь:"""


def run_one_prompt(
    model_key: str,
    model_cfg: dict,
    prompt_name: str,
    prompt: str,
    max_new_tokens: int,
):
    print("\n" + "=" * 90)
    print(f"Model key: {model_key}")
    print(f"Model name: {model_cfg['model_name']}")
    print(f"Quantization: {model_cfg['quantization_name']}")
    print(f"Prompt: {prompt_name}")
    print("=" * 90)

    try:
        result = call_ollama_chat(
            prompt=prompt,
            model=model_cfg["model_name"],
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            num_ctx=GENERATION_CONFIG.get("num_ctx", 2048),
            num_gpu=OLLAMA_NUM_GPU,
            timeout=900,
        )

        error = None
        output_text = result["text"]

        print("Output:", output_text)
        print("wall_time_sec:", round(result["wall_time_sec"], 3))
        print("load_duration_sec:", result["load_duration_sec"])
        print("prompt_tokens_per_sec:", result["prompt_tokens_per_sec"])
        print("generation_tokens_per_sec:", result["generation_tokens_per_sec"])
        print("model_process_peak_rss_mb:", result.get("model_process_peak_rss_mb"))

    except Exception as exc:
        result = {}
        error = repr(exc)
        output_text = ""

        print("ERROR:", error)

    return {
        "model_key": model_key,
        "model_name": model_cfg["model_name"],
        "display_name": model_cfg["display_name"],
        "backend_name": model_cfg["backend_name"],
        "quantization_name": model_cfg["quantization_name"],
        "source_repo": model_cfg["source_repo"],
        "artifact_family": model_cfg["artifact_family"],
        "quantization_pipeline": model_cfg["quantization_pipeline"],
        "imatrix_used": model_cfg["imatrix_used"],
        "model_role": model_cfg["model_role"],

        "requested_num_gpu": result.get("requested_num_gpu"),
        "prompt_name": prompt_name,
        "output_text": output_text,
        "error": error,

        "wall_time_sec": result.get("wall_time_sec"),
        "total_duration_sec": result.get("total_duration_sec"),
        "load_duration_sec": result.get("load_duration_sec"),

        "prompt_eval_count": result.get("prompt_eval_count"),
        "eval_count": result.get("eval_count"),
        "prompt_tokens_per_sec": result.get("prompt_tokens_per_sec"),
        "generation_tokens_per_sec": result.get("generation_tokens_per_sec"),

        "client_process_peak_rss_mb": result.get("client_process_peak_rss_mb"),
        "model_process_rss_before_mb": result.get("model_process_rss_before_mb"),
        "model_process_rss_after_mb": result.get("model_process_rss_after_mb"),
        "model_process_peak_rss_mb": result.get("model_process_peak_rss_mb"),
        "system_used_memory_peak_mb": result.get("system_used_memory_peak_mb"),
    }


def main():
    output_prefix = "smoke_gguf_matrix_isolated"
    version = next_experiment_version(RESULTS_DIR, output_prefix)
    experiment_version = f"v{version}"

    print(f"\n=== ISOLATED GGUF MATRIX SMOKE TEST {experiment_version} ===")
    print(f"Requested num_gpu: {OLLAMA_NUM_GPU}")
    print("Important: for true CPU-only mode, Ollama server should be started with CUDA_VISIBLE_DEVICES=-1.")

    rows = []

    prompts = [
        (
            "qa_uk_simple",
            qa_smoke_prompt_uk(),
            GENERATION_CONFIG.get("max_new_tokens", 32),
        ),
        (
            "belebele_uk_simple",
            belebele_smoke_prompt_uk(),
            MC_GENERATION_CONFIG.get("max_new_tokens", 4),
        ),
    ]

    print_ollama_ps("before smoke")

    for model_key in SMOKE_GGUF_MODEL_KEYS:
        model_cfg = OLLAMA_GGUF_MODELS[model_key]
        model_name = model_cfg["model_name"]

        # Make sure this specific model is not already loaded.
        stop_model(model_name)
        print_ollama_ps(f"after stopping {model_key}")

        for prompt_name, prompt, max_new_tokens in prompts:
            row = run_one_prompt(
                model_key=model_key,
                model_cfg=model_cfg,
                prompt_name=prompt_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            row["experiment_version"] = experiment_version
            rows.append(row)

            ps_after_prompt = print_ollama_ps(f"after {model_key} / {prompt_name}")
            row["ollama_ps_after_prompt"] = ps_after_prompt

        # Important: release this model before next quant.
        stop_model(model_name)
        print_ollama_ps(f"after final stop {model_key}")

    df = pd.DataFrame(rows)

    output_path = RESULTS_DIR / f"{output_prefix}_{experiment_version}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n=== ISOLATED SMOKE SUMMARY ===")
    print(
        df[
            [
                "model_key",
                "prompt_name",
                "quantization_name",
                "model_role",
                "error",
                "wall_time_sec",
                "load_duration_sec",
                "model_process_peak_rss_mb",
                "output_text",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved isolated smoke results to: {output_path}")


if __name__ == "__main__":
    main()