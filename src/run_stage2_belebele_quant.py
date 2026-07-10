import os
import re
import subprocess
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from belebele_loader import load_belebele_uk_en_subsets
from config_stage2_quant import (
    PROJECT_ROOT,
    RESULTS_DIR,
    OLLAMA_GGUF_MODELS,
    STAGE2_BELEBELE_QUANT_MODEL_KEYS,
    OLLAMA_NUM_GPU,
    RUNTIME_PROCESSOR,
    MC_GENERATION_CONFIG,
    BELEBELE_SUBSET_SIZE,
)
from eval_utils import safe_max, safe_mean
from ollama_runner import call_ollama_chat


CYR_TO_LAT = str.maketrans(
    {
        "А": "A",
        "Б": "B",
        "В": "B",
        "С": "C",
        "Д": "D",
        "а": "A",
        "б": "B",
        "в": "B",
        "с": "C",
        "д": "D",
    }
)


def make_prompt_uk(example: dict) -> str:
    return f"""Прочитай текст і вибери правильну відповідь.

Ти маєш відповісти рівно одним символом: латинською літерою A, B, C або D.
Не використовуй кириличні літери: А, Б, В, Г, Д.
Не пояснюй. Не пиши слів. Поверни тільки одну латинську літеру.

Текст:
{example["passage"]}

Питання:
{example["question"]}

A. {example["choice_a"]}
B. {example["choice_b"]}
C. {example["choice_c"]}
D. {example["choice_d"]}

Відповідь:"""


def make_prompt_en(example: dict) -> str:
    return f"""Read the passage and choose the correct answer.

You must answer with exactly one character: the Latin letter A, B, C, or D.
Do not explain. Do not write words. Return only one Latin letter.

Passage:
{example["passage"]}

Question:
{example["question"]}

A. {example["choice_a"]}
B. {example["choice_b"]}
C. {example["choice_c"]}
D. {example["choice_d"]}

Answer:"""


def extract_choice(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    cleaned = text.strip().translate(CYR_TO_LAT).upper()

    match = re.search(r"^\s*\(?\s*([ABCD])\s*\)?[\.\:]?\s*$", cleaned)
    if match:
        return match.group(1)

    patterns = [
        r"(?:ANSWER|ВІДПОВІДЬ)\s*(?:IS|Є|:)?\s*\(?\s*([ABCD])\b",
        r"(?:CORRECT\s+ANSWER\s+IS)\s*\(?\s*([ABCD])\b",
        r"(?:ПРАВИЛЬНА\s+ВІДПОВІДЬ)\s*(?:Є|:)?\s*\(?\s*([ABCD])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)

    match = re.search(r"\b([ABCD])\b", cleaned)
    if match:
        return match.group(1)

    return None


def run_live(command: list[str], check: bool = True) -> int:
    print("\n" + "=" * 100)
    print("$ " + " ".join(command))
    print("=" * 100)

    completed = subprocess.run(command, cwd=PROJECT_ROOT)

    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {completed.returncode}: {' '.join(command)}"
        )

    return completed.returncode


def ollama_pull(model_name: str) -> None:
    run_live(["ollama", "pull", model_name], check=True)


def ollama_stop(model_name: str) -> None:
    run_live(["ollama", "stop", model_name], check=False)


def ollama_ps() -> None:
    run_live(["ollama", "ps"], check=False)


def restart_ollama_server() -> None:
    print("\n[COLD START OLLAMA SERVER]")

    if os.name == "nt":
        run_live(["taskkill", "/F", "/IM", "ollama.exe"], check=False)
    else:
        run_live(["pkill", "-f", "ollama"], check=False)

    time.sleep(5)

    popen_kwargs = {
        "cwd": PROJECT_ROOT,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }

    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(["ollama", "serve"], **popen_kwargs)

    time.sleep(10)
    ollama_ps()


def stop_all_stage2_models() -> None:
    model_names = {
        OLLAMA_GGUF_MODELS[key]["model_name"]
        for key in STAGE2_BELEBELE_QUANT_MODEL_KEYS
    }

    for model_name in sorted(model_names):
        ollama_stop(model_name)

    time.sleep(3)
    ollama_ps()


def prepare_runtime_for_model() -> None:
    if os.getenv("STAGE2_COLD_START_OLLAMA", "1") == "1":
        restart_ollama_server()
    else:
        stop_all_stage2_models()


def generate_answer(prompt: str, cfg: dict) -> dict:
    max_new_tokens = MC_GENERATION_CONFIG.get("max_new_tokens", 4)

    if cfg.get("requires_long_generation_budget", False):
        max_new_tokens = max(max_new_tokens, 1024)

    return call_ollama_chat(
        prompt=prompt,
        model=cfg["model_name"],
        temperature=MC_GENERATION_CONFIG.get("temperature", 0.0),
        max_new_tokens=max_new_tokens,
        num_ctx=MC_GENERATION_CONFIG.get("num_ctx", 2048),
        num_gpu=OLLAMA_NUM_GPU,
        prompt_prefix=cfg.get("prompt_prefix", ""),
        ollama_think=cfg.get("ollama_think"),
    )


def run_eval_for_dataset(
    dataset,
    cfg: dict,
    lang: str,
    subset_name: str,
    experiment_version: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []

    for example in tqdm(
        dataset,
        desc=f"{cfg['backend_name']}-{lang}",
        ascii=True,
        dynamic_ncols=False,
        ncols=120,
    ):
        ex = dict(example)
        prompt = make_prompt_uk(ex) if lang == "uk" else make_prompt_en(ex)

        result = generate_answer(prompt, cfg)

        raw_prediction = result["text"]
        predicted_letter = extract_choice(raw_prediction)

        is_valid = predicted_letter is not None
        is_correct = int(predicted_letter == ex["correct_letter"]) if is_valid else 0

        rows.append(
            {
                "experiment_version": experiment_version,
                "example_id": ex.get("id", ""),
                "benchmark": "belebele",
                "lang": lang,
                "lang_code": ex["lang_code"],
                "subset": subset_name,

                "model_name": cfg["display_name"],
                "model_name_runtime": result.get("model"),
                "backend_name": cfg["backend_name"],
                "quantization_name": cfg["quantization_name"],
                "runtime_processor": RUNTIME_PROCESSOR,
                "requested_num_gpu": result.get("requested_num_gpu"),
                "requested_ollama_think": result.get("requested_ollama_think"),

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
                "requires_long_generation_budget": cfg.get(
                    "requires_long_generation_budget",
                    False,
                ),

                "question": ex["question"],
                "choice_a": ex["choice_a"],
                "choice_b": ex["choice_b"],
                "choice_c": ex["choice_c"],
                "choice_d": ex["choice_d"],

                "raw_prediction": raw_prediction,
                "raw_content": result.get("raw_content", ""),
                "thinking": result.get("thinking", ""),
                "has_thinking": result.get("has_thinking"),
                "clean_text_source": result.get("clean_text_source"),
                "used_thinking_fallback": result.get("used_thinking_fallback"),

                "predicted_letter": predicted_letter,
                "gold_letter": ex["correct_letter"],
                "is_valid_answer": int(is_valid),
                "is_correct": is_correct,

                "wall_time_sec": result["wall_time_sec"],
                "total_duration_sec": result["total_duration_sec"],
                "load_duration_sec": result["load_duration_sec"],

                "prompt_eval_count": result["prompt_eval_count"],
                "prompt_eval_duration_sec": result["prompt_eval_duration_sec"],
                "eval_count": result["eval_count"],
                "eval_duration_sec": result["eval_duration_sec"],

                "prompt_tokens_per_sec": result["prompt_tokens_per_sec"],
                "generation_tokens_per_sec": result["generation_tokens_per_sec"],

                "client_process_rss_before_mb": result.get("client_process_rss_before_mb"),
                "client_process_rss_after_mb": result.get("client_process_rss_after_mb"),
                "client_process_peak_rss_mb": result.get("client_process_peak_rss_mb"),

                "model_process_rss_before_mb": result.get("model_process_rss_before_mb"),
                "model_process_rss_after_mb": result.get("model_process_rss_after_mb"),
                "model_process_peak_rss_mb": result.get("model_process_peak_rss_mb"),
                "model_process_count_before": result.get("model_process_count_before"),
                "model_process_count_after": result.get("model_process_count_after"),

                "system_used_memory_before_mb": result.get("system_used_memory_before_mb"),
                "system_used_memory_after_mb": result.get("system_used_memory_after_mb"),
                "system_used_memory_peak_mb": result.get("system_used_memory_peak_mb"),
            }
        )

    df = pd.DataFrame(rows)

    summary = {
        "experiment_version": experiment_version,
        "benchmark": "belebele",

        "model_name": cfg["display_name"],
        "backend_name": cfg["backend_name"],
        "quantization_name": cfg["quantization_name"],
        "runtime_processor": RUNTIME_PROCESSOR,
        "requested_num_gpu": OLLAMA_NUM_GPU,
        "requested_ollama_think": cfg.get("ollama_think"),

        "source_repo": cfg["source_repo"],
        "artifact_family": cfg["artifact_family"],
        "quantization_pipeline": cfg["quantization_pipeline"],
        "imatrix_used": cfg["imatrix_used"],
        "model_role": cfg["model_role"],
        "base_model_family": cfg.get("base_model_family", ""),
        "reasoning_mode": cfg.get("reasoning_mode", ""),
        "model_size_class": cfg.get("model_size_class", ""),
        "architecture_note": cfg.get("architecture_note", ""),

        "lang": lang,
        "subset": subset_name,
        "n_examples": len(df),

        "accuracy": safe_mean(df, "is_correct", digits=4),
        "invalid_answer_rate": (
            round(float(1.0 - df["is_valid_answer"].mean()), 4)
            if len(df)
            else 0.0
        ),

        "avg_wall_time_sec": safe_mean(df, "wall_time_sec", digits=4),
        "avg_total_duration_sec": safe_mean(df, "total_duration_sec", digits=4),
        "avg_load_duration_sec": safe_mean(df, "load_duration_sec", digits=4),

        "avg_prompt_eval_count": safe_mean(df, "prompt_eval_count", digits=2),
        "avg_eval_count": safe_mean(df, "eval_count", digits=2),

        "avg_prompt_tokens_per_sec": safe_mean(df, "prompt_tokens_per_sec", digits=4),
        "avg_generation_tokens_per_sec": safe_mean(
            df,
            "generation_tokens_per_sec",
            digits=4,
        ),

        "avg_client_process_peak_rss_mb": safe_mean(
            df,
            "client_process_peak_rss_mb",
            digits=2,
        ),
        "avg_model_process_peak_rss_mb": safe_mean(
            df,
            "model_process_peak_rss_mb",
            digits=2,
        ),
        "max_model_process_peak_rss_mb": safe_max(
            df,
            "model_process_peak_rss_mb",
            digits=2,
        ),
        "avg_model_process_rss_before_mb": safe_mean(
            df,
            "model_process_rss_before_mb",
            digits=2,
        ),
        "avg_model_process_rss_after_mb": safe_mean(
            df,
            "model_process_rss_after_mb",
            digits=2,
        ),
        "avg_system_used_memory_peak_mb": safe_mean(
            df,
            "system_used_memory_peak_mb",
            digits=2,
        ),
        "max_system_used_memory_peak_mb": safe_max(
            df,
            "system_used_memory_peak_mb",
            digits=2,
        ),
        "thinking_rate": safe_mean(df, "has_thinking", digits=4),
        "thinking_fallback_rate": safe_mean(df, "used_thinking_fallback", digits=4),
    }

    return df, summary


def save_manifest(output_dir, experiment_version: str) -> None:
    rows = []

    for i, key in enumerate(STAGE2_BELEBELE_QUANT_MODEL_KEYS, start=1):
        cfg = OLLAMA_GGUF_MODELS[key]
        rows.append(
            {
                "run_order": i,
                "experiment_version": experiment_version,
                "model_key": key,
                **cfg,
            }
        )

    path = output_dir / f"stage2_belebele_quant_manifest_{experiment_version}.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {path}")


def build_output_paths(output_dir, cfg: dict, experiment_version: str) -> tuple:
    details_path = (
        output_dir
        / f"stage2_belebele_details_{cfg['backend_name']}_{BELEBELE_SUBSET_SIZE}_{experiment_version}.csv"
    )
    summary_path = (
        output_dir
        / f"stage2_belebele_summary_{cfg['backend_name']}_{BELEBELE_SUBSET_SIZE}_{experiment_version}.csv"
    )
    return details_path, summary_path


def run_model(model_key: str, output_dir, experiment_version: str) -> None:
    cfg = OLLAMA_GGUF_MODELS[model_key]

    details_path, summary_path = build_output_paths(
        output_dir=output_dir,
        cfg=cfg,
        experiment_version=experiment_version,
    )

    if details_path.exists() and summary_path.exists():
        print(f"\n[SKIP] Existing results found for {cfg['backend_name']}")
        print(f"[SKIP] {details_path}")
        print(f"[SKIP] {summary_path}")
        return

    print("\n" + "#" * 100)
    print(f"# STAGE2 BELEBELE QUANT: {model_key}")
    print(f"# Model: {cfg['model_name']}")
    print(f"# Display: {cfg['display_name']}")
    print(f"# Quantization: {cfg['quantization_name']}")
    print(f"# Family: {cfg.get('base_model_family')}")
    print(f"# Cold start Ollama: {os.getenv('STAGE2_COLD_START_OLLAMA', '1')}")
    print("#" * 100)

    prepare_runtime_for_model()

    if os.getenv("STAGE2_NO_PULL", "1") != "1":
        ollama_pull(cfg["model_name"])

    uk_ds, en_ds = load_belebele_uk_en_subsets(BELEBELE_SUBSET_SIZE)

    uk_df, uk_summary = run_eval_for_dataset(
        dataset=uk_ds,
        cfg=cfg,
        lang="uk",
        subset_name=f"belebele_ukr_Cyrl_{BELEBELE_SUBSET_SIZE}",
        experiment_version=experiment_version,
    )

    en_df, en_summary = run_eval_for_dataset(
        dataset=en_ds,
        cfg=cfg,
        lang="en",
        subset_name=f"belebele_eng_Latn_{BELEBELE_SUBSET_SIZE}",
        experiment_version=experiment_version,
    )

    details = pd.concat([uk_df, en_df], ignore_index=True)
    summary = pd.DataFrame([uk_summary, en_summary])

    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {details_path}")
    print(f"[SAVED] {summary_path}")

    ollama_stop(cfg["model_name"])
    time.sleep(3)


def main() -> None:
    experiment_version = os.getenv("EXPERIMENT_VERSION", "stage2_belebele_quant_200_v1")
    output_dir = RESULTS_DIR / experiment_version
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== STAGE2 BELEBELE QUANT ===")
    print(f"Version: {experiment_version}")
    print(f"Output dir: {output_dir}")
    print(f"Models: {len(STAGE2_BELEBELE_QUANT_MODEL_KEYS)}")
    print(f"BELEBELE size: {BELEBELE_SUBSET_SIZE}")
    print(f"Cold start Ollama: {os.getenv('STAGE2_COLD_START_OLLAMA', '1')}")
    print(f"Pull models: {os.getenv('STAGE2_NO_PULL', '1') != '1'}")

    save_manifest(output_dir, experiment_version)

    for model_key in STAGE2_BELEBELE_QUANT_MODEL_KEYS:
        run_model(model_key, output_dir, experiment_version)

    print("\nDone.")


if __name__ == "__main__":
    main()