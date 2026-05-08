import re
import string
from collections import Counter

import pandas as pd
from tqdm import tqdm

from dataset_loaders import (
    load_ua_squad_validation_subset,
    load_en_squad_validation_subset,
)
from hf_master_runner import call_hf_master_chat
from config import RESULTS_DIR, UA_QA_SUBSET_SIZE, EN_QA_SUBSET_SIZE


MASTER_MODEL_DISPLAY_NAME = "Phi-4-mini-instruct-master"
MASTER_BACKEND_NAME = "transformers_cpu_master"
MASTER_QUANTIZATION_NAME = "higher_precision_auto"
MASTER_RUNTIME_PROCESSOR = "CPU"


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


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation + "«»„“”’…")
    return text


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(
    prediction: str,
    ground_truths: list[str],
    metric_fn,
):
    cleaned = [gt for gt in ground_truths if isinstance(gt, str) and gt.strip()]

    if not cleaned:
        return None

    return max(metric_fn(prediction, gt) for gt in cleaned)


def make_prompt_uk(context: str, question: str) -> str:
    return f"""Прочитай контекст і дай коротку точну відповідь на запитання.
Відповідай лише самою відповіддю, без пояснень.

Контекст:
{context}

Запитання:
{question}

Відповідь:"""


def make_prompt_en(context: str, question: str) -> str:
    return f"""Read the context and answer the question briefly and precisely.
Return only the answer, without explanation.

Context:
{context}

Question:
{question}

Answer:"""


def generate_answer(prompt: str) -> dict:
    return call_hf_master_chat(
        prompt=prompt,
        max_new_tokens=32,
    )


def warmup_model():
    print("\n[WARMUP] Loading master model before QA benchmark...")

    warmup_prompt = """Прочитай контекст і дай коротку точну відповідь на запитання.
Відповідай лише самою відповіддю, без пояснень.

Контекст:
Нормандія розташована у Франції.

Запитання:
У якій країні розташована Нормандія?

Відповідь:"""

    result = generate_answer(warmup_prompt)

    print("[WARMUP] Output:", result["text"])
    print("[WARMUP] wall_time_sec:", round(result["wall_time_sec"], 3))
    print("[WARMUP] load_duration_sec:", round(result["load_duration_sec"], 3))
    print("[WARMUP] model_dtype:", result.get("model_dtype"))
    print("[WARMUP] rss_mb_after_load:", result.get("rss_mb_after_load"))
    print("[WARMUP] model_process_peak_rss_mb:", result.get("model_process_peak_rss_mb"))


def safe_mean(df: pd.DataFrame, column: str, digits: int = 4):
    if column not in df.columns or len(df) == 0:
        return 0.0

    values = df[column].dropna()

    if values.empty:
        return 0.0

    return round(float(values.mean()), digits)


def safe_max(df: pd.DataFrame, column: str, digits: int = 4):
    if column not in df.columns or len(df) == 0:
        return 0.0

    values = df[column].dropna()

    if values.empty:
        return 0.0

    return round(float(values.max()), digits)


def first_non_null(df: pd.DataFrame, column: str):
    if column not in df.columns or len(df) == 0:
        return None

    values = df[column].dropna()

    if values.empty:
        return None

    return values.iloc[0]


def run_eval(
    dataset,
    lang: str,
    subset_name: str,
    experiment_version: str,
):
    rows = []
    skipped_no_answers = 0

    for example in tqdm(
        dataset,
        desc=f"{lang}-{subset_name}",
        ascii=True,
        dynamic_ncols=False,
        ncols=120,
    ):
        context = example["context"]
        question = example["question"]

        raw_answers = example["answers"]["text"]
        gold_answers = [a for a in raw_answers if isinstance(a, str) and a.strip()]

        if not gold_answers:
            skipped_no_answers += 1
            continue

        prompt = make_prompt_uk(context, question) if lang == "uk" else make_prompt_en(
            context,
            question,
        )

        result = generate_answer(prompt)
        prediction = result["text"]

        em = metric_max_over_ground_truths(prediction, gold_answers, exact_match)
        f1 = metric_max_over_ground_truths(prediction, gold_answers, f1_score)

        prompt_token_count = result.get("prompt_token_count")
        generated_token_count = result.get("generated_token_count")

        rows.append(
            {
                "experiment_version": experiment_version,

                "example_id": example.get("id", ""),
                "lang": lang,
                "subset": subset_name,

                "model_name": MASTER_MODEL_DISPLAY_NAME,
                "model_id": result.get("model_id"),
                "model_dtype": result.get("model_dtype"),

                "backend_name": MASTER_BACKEND_NAME,
                "quantization_name": MASTER_QUANTIZATION_NAME,
                "runtime_processor": MASTER_RUNTIME_PROCESSOR,
                "requested_num_gpu": None,

                "question": question,
                "prediction": prediction,
                "gold_answers": " | ".join(gold_answers),

                "exact_match": em if em is not None else 0,
                "f1": f1 if f1 is not None else 0.0,

                "wall_time_sec": result["wall_time_sec"],
                "total_duration_sec": result["total_duration_sec"],
                "load_duration_sec": result["load_duration_sec"],

                "prompt_token_count": prompt_token_count,
                "generated_token_count": generated_token_count,

                # Aliases for easier comparison with Ollama result schema.
                "prompt_eval_count": prompt_token_count,
                "eval_count": generated_token_count,

                "effective_total_tokens_per_sec": result.get(
                    "effective_total_tokens_per_sec"
                ),
                "effective_generation_tokens_per_sec": result.get(
                    "effective_generation_tokens_per_sec"
                ),

                "rss_mb_after_load": result.get("rss_mb_after_load"),

                "model_process_rss_before_mb": result.get(
                    "model_process_rss_before_mb"
                ),
                "model_process_rss_after_mb": result.get(
                    "model_process_rss_after_mb"
                ),
                "model_process_peak_rss_mb": result.get(
                    "model_process_peak_rss_mb"
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

    summary = {
        "experiment_version": experiment_version,

        "model_name": MASTER_MODEL_DISPLAY_NAME,
        "model_id": first_non_null(df, "model_id"),
        "model_dtype": first_non_null(df, "model_dtype"),

        "backend_name": MASTER_BACKEND_NAME,
        "quantization_name": MASTER_QUANTIZATION_NAME,
        "runtime_processor": MASTER_RUNTIME_PROCESSOR,
        "requested_num_gpu": None,

        "lang": lang,
        "subset": subset_name,
        "n_examples": len(df),
        "skipped_no_answers": skipped_no_answers,

        "avg_em": safe_mean(df, "exact_match", digits=4),
        "avg_f1": safe_mean(df, "f1", digits=4),

        "avg_wall_time_sec": safe_mean(df, "wall_time_sec", digits=4),
        "avg_total_duration_sec": safe_mean(df, "total_duration_sec", digits=4),
        "avg_load_duration_sec": safe_mean(df, "load_duration_sec", digits=4),

        "avg_prompt_token_count": safe_mean(df, "prompt_token_count", digits=2),
        "avg_generated_token_count": safe_mean(df, "generated_token_count", digits=2),

        "avg_prompt_eval_count": safe_mean(df, "prompt_eval_count", digits=2),
        "avg_eval_count": safe_mean(df, "eval_count", digits=2),

        "avg_effective_total_tokens_per_sec": safe_mean(
            df,
            "effective_total_tokens_per_sec",
            digits=4,
        ),
        "avg_effective_generation_tokens_per_sec": safe_mean(
            df,
            "effective_generation_tokens_per_sec",
            digits=4,
        ),

        "avg_rss_mb_after_load": safe_mean(df, "rss_mb_after_load", digits=2),

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
    }

    return df, summary


def main():
    details_prefix = (
        f"pilot_qa_details_{MASTER_BACKEND_NAME}_ua{UA_QA_SUBSET_SIZE}_en{EN_QA_SUBSET_SIZE}"
    )

    version_num = next_experiment_version(RESULTS_DIR, details_prefix)
    experiment_version = f"v{version_num}"

    print(f"\n=== MASTER QA EXPERIMENT VERSION: {experiment_version} ===")
    print(f"Backend: {MASTER_BACKEND_NAME}")
    print(f"Quantization: {MASTER_QUANTIZATION_NAME}")
    print(f"Runtime processor: {MASTER_RUNTIME_PROCESSOR}")
    print(f"UA subset size: {UA_QA_SUBSET_SIZE}")
    print(f"EN subset size: {EN_QA_SUBSET_SIZE}")

    warmup_model()

    ua_ds = load_ua_squad_validation_subset(UA_QA_SUBSET_SIZE)
    en_ds = load_en_squad_validation_subset(EN_QA_SUBSET_SIZE)

    ua_df, ua_summary = run_eval(
        ua_ds,
        lang="uk",
        subset_name=f"ua_squad_{UA_QA_SUBSET_SIZE}",
        experiment_version=experiment_version,
    )

    en_df, en_summary = run_eval(
        en_ds,
        lang="en",
        subset_name=f"squad_{EN_QA_SUBSET_SIZE}",
        experiment_version=experiment_version,
    )

    details = pd.concat([ua_df, en_df], ignore_index=True)
    summary_df = pd.DataFrame([ua_summary, en_summary])

    details_path = (
        RESULTS_DIR
        / f"pilot_qa_details_{MASTER_BACKEND_NAME}_ua{UA_QA_SUBSET_SIZE}_en{EN_QA_SUBSET_SIZE}_{experiment_version}.csv"
    )

    summary_path = (
        RESULTS_DIR
        / f"pilot_qa_summary_{MASTER_BACKEND_NAME}_ua{UA_QA_SUBSET_SIZE}_en{EN_QA_SUBSET_SIZE}_{experiment_version}.csv"
    )

    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== MASTER QA SUMMARY ===")
    print(summary_df)
    print(f"\nSaved details to: {details_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()