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
)
from ollama_runner import call_ollama_chat


# Normalization note:
# The benchmark expects Latin answer labels: A, B, C, D.
# In Ukrainian prompts, the model may sometimes return Cyrillic-looking labels
# such as "Б" for option B, "В" as a visual substitute for Latin B,
# "С" as a visual substitute for Latin C, or "Д" for option D.
# We normalize these labels to avoid counting formatting/script variants
# as invalid answers when the intended option is clear.
#
# This normalization affects only the predicted multiple-choice label,
# not the question text, passage, or answer options.
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


def next_experiment_version(results_dir, prefix: str) -> int:
    existing_versions = []

    for path in results_dir.glob(f"{prefix}_v*.csv"):
        stem = path.stem

        try:
            version_part = stem.rsplit("_v", 1)[1]
            existing_versions.append(int(version_part))
        except (IndexError, ValueError):
            continue

    if not existing_versions:
        return 1

    return max(existing_versions) + 1


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

    # Прямі короткі відповіді: A, A., A), (A)
    match = re.search(r"^\s*\(?\s*([ABCD])\s*\)?[\.\:]?\s*$", cleaned)
    if match:
        return match.group(1)

    # Відповідь: A / Answer: A / Correct answer is A
    patterns = [
        r"(?:ANSWER|ВІДПОВІДЬ)\s*(?:IS|Є|:)?\s*\(?\s*([ABCD])\b",
        r"(?:CORRECT\s+ANSWER\s+IS)\s*\(?\s*([ABCD])\b",
        r"(?:ПРАВИЛЬНА\s+ВІДПОВІДЬ)\s*(?:Є|:)?\s*\(?\s*([ABCD])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)

    # Fallback: перша ізольована літера A/B/C/D
    match = re.search(r"\b([ABCD])\b", cleaned)
    if match:
        return match.group(1)

    return None


def generate_answer(prompt: str) -> dict:
    return call_ollama_chat(
        prompt=prompt,
        model=PRIMARY_MODEL_NAME,
        temperature=MC_GENERATION_CONFIG.get("temperature", 0.0),
        max_new_tokens=MC_GENERATION_CONFIG.get("max_new_tokens", 4),
        num_ctx=MC_GENERATION_CONFIG.get("num_ctx", 2048),
        num_gpu=OLLAMA_NUM_GPU,
    )


def run_eval(
    dataset,
    lang: str,
    subset_name: str,
    experiment_version: str,
):
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
                "requested_num_gpu": result.get("requested_num_gpu"),

                "question": ex["question"],
                "choice_a": ex["choice_a"],
                "choice_b": ex["choice_b"],
                "choice_c": ex["choice_c"],
                "choice_d": ex["choice_d"],

                "raw_prediction": raw_prediction,
                "predicted_letter": predicted_letter,
                "gold_letter": ex["correct_letter"],
                "is_valid_answer": int(is_valid),
                "is_correct": is_correct,

                "wall_time_sec": result["wall_time_sec"],
                "total_duration_sec": result["total_duration_sec"],
                "load_duration_sec": result["load_duration_sec"],

                "prompt_eval_count": result["prompt_eval_count"],
                "eval_count": result["eval_count"],
                "prompt_tokens_per_sec": result["prompt_tokens_per_sec"],
                "generation_tokens_per_sec": result["generation_tokens_per_sec"],

                "client_process_rss_before_mb": result.get(
                    "client_process_rss_before_mb"
                ),
                "client_process_rss_after_mb": result.get(
                    "client_process_rss_after_mb"
                ),
                "client_process_peak_rss_mb": result.get(
                    "client_process_peak_rss_mb"
                ),

                "model_process_rss_before_mb": result.get(
                    "model_process_rss_before_mb"
                ),
                "model_process_rss_after_mb": result.get(
                    "model_process_rss_after_mb"
                ),
                "model_process_peak_rss_mb": result.get(
                    "model_process_peak_rss_mb"
                ),
                "model_process_count_before": result.get(
                    "model_process_count_before"
                ),
                "model_process_count_after": result.get(
                    "model_process_count_after"
                ),

                "system_used_memory_before_mb": result.get(
                    "system_used_memory_before_mb"
                ),
                "system_used_memory_after_mb": result.get(
                    "system_used_memory_after_mb"
                ),
                "system_used_memory_peak_mb": result.get(
                    "system_used_memory_peak_mb"
                ),
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
        "requested_num_gpu": requested_num_gpu,
        "lang": lang,
        "subset": subset_name,
        "n_examples": len(df),

        "accuracy": round(float(df["is_correct"].mean()), 4) if len(df) else 0.0,
        "invalid_answer_rate": (
            round(float(1.0 - df["is_valid_answer"].mean()), 4)
            if len(df)
            else 0.0
        ),

        "avg_wall_time_sec": (
            round(float(df["wall_time_sec"].mean()), 4) if len(df) else 0.0
        ),
        "avg_total_duration_sec": (
            round(float(df["total_duration_sec"].mean()), 4) if len(df) else 0.0
        ),
        "avg_load_duration_sec": (
            round(float(df["load_duration_sec"].mean()), 4) if len(df) else 0.0
        ),

        "avg_prompt_tokens_per_sec": (
            round(float(df["prompt_tokens_per_sec"].dropna().mean()), 4)
            if len(df)
            else 0.0
        ),
        "avg_generation_tokens_per_sec": (
            round(float(df["generation_tokens_per_sec"].dropna().mean()), 4)
            if len(df)
            else 0.0
        ),

        "avg_model_process_peak_rss_mb": (
            round(float(df["model_process_peak_rss_mb"].dropna().mean()), 2)
            if len(df)
            else 0.0
        ),
        "max_model_process_peak_rss_mb": (
            round(float(df["model_process_peak_rss_mb"].dropna().max()), 2)
            if len(df)
            else 0.0
        ),

        "avg_system_used_memory_peak_mb": (
            round(float(df["system_used_memory_peak_mb"].dropna().mean()), 2)
            if len(df)
            else 0.0
        ),
        "max_system_used_memory_peak_mb": (
            round(float(df["system_used_memory_peak_mb"].dropna().max()), 2)
            if len(df)
            else 0.0
        ),
    }

    return df, summary


def main():
    details_prefix = (
        f"pilot_belebele_details_{BACKEND_NAME}_{BELEBELE_SUBSET_SIZE}"
    )
    version_num = next_experiment_version(RESULTS_DIR, details_prefix)
    experiment_version = f"v{version_num}"

    print(f"\n=== EXPERIMENT VERSION: {experiment_version} ===")
    print(f"Backend: {BACKEND_NAME}")
    print(f"Subset size: {BELEBELE_SUBSET_SIZE}")
    print(f"Requested num_gpu: {OLLAMA_NUM_GPU}")

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
        RESULTS_DIR
        / f"pilot_belebele_details_{BACKEND_NAME}_{BELEBELE_SUBSET_SIZE}_{experiment_version}.csv"
    )

    summary_path = (
        RESULTS_DIR
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