import re

import pandas as pd
from tqdm import tqdm

from belebele_loader import load_belebele_uk_en_subsets
from config import (
    RESULTS_DIR,
    PRIMARY_MODEL_NAME,
    PRIMARY_MODEL_DISPLAY_NAME,
    BACKEND_NAME,
    OLLAMA_NUM_GPU,
    BELEBELE_SUBSET_SIZE,
    MC_GENERATION_CONFIG,
    QUANTIZATION_NAME,
    RUNTIME_PROCESSOR,
    MODEL_SOURCE_REPO,
    ARTIFACT_FAMILY,
    QUANTIZATION_PIPELINE,
    IMATRIX_USED,
    MODEL_ROLE,
    PROMPT_PREFIX,
    OLLAMA_THINK,
    REASONING_MODE,
)
from eval_utils import (
    make_experiment_output_dir,
    resolve_experiment_version,
    safe_max,
    safe_mean,
)
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


def generate_answer(prompt: str) -> dict:
    max_new_tokens = MC_GENERATION_CONFIG.get("max_new_tokens", 4)

    if REASONING_MODE in {"think", "reasoning_only", "default_or_hybrid"}:
        max_new_tokens = max(max_new_tokens, 1024)

    return call_ollama_chat(
        prompt=prompt,
        model=PRIMARY_MODEL_NAME,
        temperature=MC_GENERATION_CONFIG.get("temperature", 0.0),
        max_new_tokens=max_new_tokens,
        num_ctx=MC_GENERATION_CONFIG.get("num_ctx", 2048),
        num_gpu=OLLAMA_NUM_GPU,
        prompt_prefix=PROMPT_PREFIX,
        ollama_think=OLLAMA_THINK,
    )


def run_eval(dataset, lang: str, subset_name: str, experiment_version: str):
    rows = []

    for example in tqdm(
        dataset,
        desc=f"{lang}-{subset_name}",
        ascii=True,
        dynamic_ncols=False,
        ncols=120,
    ):
        ex = dict(example)

        prompt = make_prompt_uk(ex) if lang == "uk" else make_prompt_en(ex)

        result = generate_answer(prompt)

        raw_prediction = result["text"]
        predicted_letter = extract_choice(raw_prediction)

        is_valid = predicted_letter is not None
        is_correct = int(predicted_letter == ex["correct_letter"]) if is_valid else 0

        rows.append(
            {
                "experiment_version": experiment_version,
                "example_id": ex.get("id", ""),
                "lang": lang,
                "lang_code": ex["lang_code"],
                "subset": subset_name,
                "model_name": PRIMARY_MODEL_DISPLAY_NAME,
                "backend_name": BACKEND_NAME,
                "quantization_name": QUANTIZATION_NAME,
                "runtime_processor": RUNTIME_PROCESSOR,
                "requested_num_gpu": result.get("requested_num_gpu"),
                "requested_ollama_think": result.get("requested_ollama_think"),
                "source_repo": MODEL_SOURCE_REPO,
                "artifact_family": ARTIFACT_FAMILY,
                "quantization_pipeline": QUANTIZATION_PIPELINE,
                "imatrix_used": IMATRIX_USED,
                "model_role": MODEL_ROLE,
                "reasoning_mode": REASONING_MODE,
                "prompt_prefix": PROMPT_PREFIX,
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

    requested_num_gpu = (
        int(df["requested_num_gpu"].dropna().iloc[0])
        if len(df)
        and "requested_num_gpu" in df.columns
        and not df["requested_num_gpu"].dropna().empty
        else None
    )

    summary = {
        "experiment_version": experiment_version,
        "model_name": PRIMARY_MODEL_DISPLAY_NAME,
        "backend_name": BACKEND_NAME,
        "quantization_name": QUANTIZATION_NAME,
        "runtime_processor": RUNTIME_PROCESSOR,
        "requested_num_gpu": requested_num_gpu,
        "requested_ollama_think": OLLAMA_THINK,
        "source_repo": MODEL_SOURCE_REPO,
        "artifact_family": ARTIFACT_FAMILY,
        "quantization_pipeline": QUANTIZATION_PIPELINE,
        "imatrix_used": IMATRIX_USED,
        "model_role": MODEL_ROLE,
        "reasoning_mode": REASONING_MODE,
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
        "avg_prompt_tokens_per_sec": safe_mean(df, "prompt_tokens_per_sec", digits=4),
        "avg_generation_tokens_per_sec": safe_mean(df, "generation_tokens_per_sec", digits=4),
        "avg_model_process_peak_rss_mb": safe_mean(df, "model_process_peak_rss_mb", digits=2),
        "max_model_process_peak_rss_mb": safe_max(df, "model_process_peak_rss_mb", digits=2),
        "avg_system_used_memory_peak_mb": safe_mean(df, "system_used_memory_peak_mb", digits=2),
        "max_system_used_memory_peak_mb": safe_max(df, "system_used_memory_peak_mb", digits=2),
        "thinking_rate": safe_mean(df, "has_thinking", digits=4),
        "thinking_fallback_rate": safe_mean(df, "used_thinking_fallback", digits=4),
    }

    return df, summary


def main():
    details_prefix = (
        f"pilot_belebele_details_{BACKEND_NAME}_{BELEBELE_SUBSET_SIZE}"
    )

    experiment_version = resolve_experiment_version(RESULTS_DIR, details_prefix)
    output_dir = make_experiment_output_dir(RESULTS_DIR, experiment_version)

    print(f"\n=== BELEBELE EXPERIMENT VERSION: {experiment_version} ===")
    print(f"Output dir: {output_dir}")
    print(f"Model name: {PRIMARY_MODEL_NAME}")
    print(f"Display name: {PRIMARY_MODEL_DISPLAY_NAME}")
    print(f"Backend: {BACKEND_NAME}")
    print(f"Quantization: {QUANTIZATION_NAME}")
    print(f"Runtime processor: {RUNTIME_PROCESSOR}")
    print(f"Requested num_gpu: {OLLAMA_NUM_GPU}")
    print(f"Requested ollama think: {OLLAMA_THINK}")
    print(f"Reasoning mode: {REASONING_MODE}")
    print(f"Prompt prefix: {repr(PROMPT_PREFIX)}")
    print(f"Source repo: {MODEL_SOURCE_REPO}")
    print(f"Artifact family: {ARTIFACT_FAMILY}")
    print(f"Quantization pipeline: {QUANTIZATION_PIPELINE}")
    print(f"Imatrix used: {IMATRIX_USED}")
    print(f"Model role: {MODEL_ROLE}")
    print(f"Subset size: {BELEBELE_SUBSET_SIZE}")

    uk_ds, en_ds = load_belebele_uk_en_subsets(BELEBELE_SUBSET_SIZE)

    uk_df, uk_summary = run_eval(
        uk_ds,
        lang="uk",
        subset_name=f"belebele_ukr_Cyrl_{BELEBELE_SUBSET_SIZE}",
        experiment_version=experiment_version,
    )

    en_df, en_summary = run_eval(
        en_ds,
        lang="en",
        subset_name=f"belebele_eng_Latn_{BELEBELE_SUBSET_SIZE}",
        experiment_version=experiment_version,
    )

    details = pd.concat([uk_df, en_df], ignore_index=True)
    summary_df = pd.DataFrame([uk_summary, en_summary])

    details_path = (
        output_dir
        / f"pilot_belebele_details_{BACKEND_NAME}_{BELEBELE_SUBSET_SIZE}_{experiment_version}.csv"
    )

    summary_path = (
        output_dir
        / f"pilot_belebele_summary_{BACKEND_NAME}_{BELEBELE_SUBSET_SIZE}_{experiment_version}.csv"
    )

    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== BELEBELE SUMMARY ===")
    print(summary_df)
    print(f"\nSaved details to: {details_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()