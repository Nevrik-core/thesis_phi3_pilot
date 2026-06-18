import re
import string
from collections import Counter

import pandas as pd
from tqdm import tqdm

from config import (
    RESULTS_DIR,
    PRIMARY_MODEL_NAME,
    PRIMARY_MODEL_DISPLAY_NAME,
    BACKEND_NAME,
    OLLAMA_NUM_GPU,
    UA_QA_SUBSET_SIZE,
    EN_QA_SUBSET_SIZE,
    GENERATION_CONFIG,
    QUANTIZATION_NAME,
    RUNTIME_PROCESSOR,
    MODEL_SOURCE_REPO,
    ARTIFACT_FAMILY,
    QUANTIZATION_PIPELINE,
    IMATRIX_USED,
    MODEL_ROLE,
)
from dataset_loaders import (
    load_ua_squad_validation_subset,
    load_en_squad_validation_subset,
)
from eval_utils import (
    make_experiment_output_dir,
    resolve_experiment_version,
    safe_max,
    safe_mean,
)
from ollama_runner import call_ollama_chat


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
    return call_ollama_chat(
        prompt=prompt,
        model=PRIMARY_MODEL_NAME,
        temperature=GENERATION_CONFIG.get("temperature", 0.0),
        max_new_tokens=GENERATION_CONFIG.get("max_new_tokens", 32),
        num_ctx=GENERATION_CONFIG.get("num_ctx", 2048),
        num_gpu=OLLAMA_NUM_GPU,
    )


def warmup_model():
    print("\n[WARMUP] Loading Ollama model before QA benchmark...")

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
    print("[WARMUP] load_duration_sec:", result["load_duration_sec"])
    print("[WARMUP] model_process_peak_rss_mb:", result.get("model_process_peak_rss_mb"))


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

        rows.append(
            {
                "experiment_version": experiment_version,

                "example_id": example.get("id", ""),
                "lang": lang,
                "subset": subset_name,

                "model_name": PRIMARY_MODEL_DISPLAY_NAME,
                "backend_name": BACKEND_NAME,
                "quantization_name": QUANTIZATION_NAME,
                "runtime_processor": RUNTIME_PROCESSOR,
                "requested_num_gpu": result.get("requested_num_gpu"),

                "source_repo": MODEL_SOURCE_REPO,
                "artifact_family": ARTIFACT_FAMILY,
                "quantization_pipeline": QUANTIZATION_PIPELINE,
                "imatrix_used": IMATRIX_USED,
                "model_role": MODEL_ROLE,

                "question": question,
                "prediction": prediction,
                "gold_answers": " | ".join(gold_answers),

                "exact_match": em if em is not None else 0,
                "f1": f1 if f1 is not None else 0.0,

                "wall_time_sec": result["wall_time_sec"],
                "total_duration_sec": result["total_duration_sec"],
                "load_duration_sec": result["load_duration_sec"],

                "prompt_eval_count": result["prompt_eval_count"],
                "prompt_eval_duration_sec": result["prompt_eval_duration_sec"],
                "eval_count": result["eval_count"],
                "eval_duration_sec": result["eval_duration_sec"],

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
        "quantization_name": QUANTIZATION_NAME,
        "runtime_processor": RUNTIME_PROCESSOR,
        "requested_num_gpu": requested_num_gpu,

        "source_repo": MODEL_SOURCE_REPO,
        "artifact_family": ARTIFACT_FAMILY,
        "quantization_pipeline": QUANTIZATION_PIPELINE,
        "imatrix_used": IMATRIX_USED,
        "model_role": MODEL_ROLE,

        "lang": lang,
        "subset": subset_name,
        "n_examples": len(df),
        "skipped_no_answers": skipped_no_answers,

        "avg_em": safe_mean(df, "exact_match", digits=4),
        "avg_f1": safe_mean(df, "f1", digits=4),

        "avg_wall_time_sec": safe_mean(df, "wall_time_sec", digits=4),
        "avg_total_duration_sec": safe_mean(df, "total_duration_sec", digits=4),
        "avg_load_duration_sec": safe_mean(df, "load_duration_sec", digits=4),

        "avg_prompt_eval_count": safe_mean(df, "prompt_eval_count", digits=2),
        "avg_eval_count": safe_mean(df, "eval_count", digits=2),

        "avg_prompt_tokens_per_sec": safe_mean(
            df,
            "prompt_tokens_per_sec",
            digits=4,
        ),
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
        f"pilot_qa_details_{BACKEND_NAME}_ua{UA_QA_SUBSET_SIZE}_en{EN_QA_SUBSET_SIZE}"
    )

    experiment_version = resolve_experiment_version(RESULTS_DIR, details_prefix)
    output_dir = make_experiment_output_dir(RESULTS_DIR, experiment_version)

    print(f"\n=== QA EXPERIMENT VERSION: {experiment_version} ===")
    print(f"Output dir: {output_dir}")
    print(f"Model name: {PRIMARY_MODEL_NAME}")
    print(f"Display name: {PRIMARY_MODEL_DISPLAY_NAME}")
    print(f"Backend: {BACKEND_NAME}")
    print(f"Quantization: {QUANTIZATION_NAME}")
    print(f"Runtime processor: {RUNTIME_PROCESSOR}")
    print(f"Requested num_gpu: {OLLAMA_NUM_GPU}")
    print(f"Source repo: {MODEL_SOURCE_REPO}")
    print(f"Artifact family: {ARTIFACT_FAMILY}")
    print(f"Quantization pipeline: {QUANTIZATION_PIPELINE}")
    print(f"Imatrix used: {IMATRIX_USED}")
    print(f"Model role: {MODEL_ROLE}")
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
        output_dir
        / f"pilot_qa_details_{BACKEND_NAME}_ua{UA_QA_SUBSET_SIZE}_en{EN_QA_SUBSET_SIZE}_{experiment_version}.csv"
    )

    summary_path = (
        output_dir
        / f"pilot_qa_summary_{BACKEND_NAME}_ua{UA_QA_SUBSET_SIZE}_en{EN_QA_SUBSET_SIZE}_{experiment_version}.csv"
    )

    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== QA SUMMARY ===")
    print(summary_df)
    print(f"\nSaved details to: {details_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()