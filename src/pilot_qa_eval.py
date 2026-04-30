import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd
import psutil
from tqdm import tqdm

from config import (
    RESULTS_DIR,
    PRIMARY_MODEL_NAME,
    PRIMARY_MODEL_DISPLAY_NAME,
    BACKEND_NAME,
    UA_QA_SUBSET_SIZE,
    EN_QA_SUBSET_SIZE,
    GENERATION_CONFIG,
)
from dataset_loaders import (
    load_ua_squad_validation_subset,
    load_en_squad_validation_subset,
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


def metric_max_over_ground_truths(prediction: str, ground_truths: list[str], metric_fn):
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


def get_memory_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def generate_answer(prompt: str) -> dict:
    return call_ollama_chat(
        prompt=prompt,
        model=PRIMARY_MODEL_NAME,
        temperature=GENERATION_CONFIG.get("temperature", 0.0),
        max_new_tokens=GENERATION_CONFIG.get("max_new_tokens", 32),
        num_ctx=GENERATION_CONFIG.get("num_ctx", 2048),
    )


def run_eval(dataset, lang: str, subset_name: str):
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

        prompt = make_prompt_uk(context, question) if lang == "uk" else make_prompt_en(context, question)

        mem_before = get_memory_mb()
        result = generate_answer(prompt)
        mem_after = get_memory_mb()

        prediction = result["text"]

        em = metric_max_over_ground_truths(prediction, gold_answers, exact_match)
        f1 = metric_max_over_ground_truths(prediction, gold_answers, f1_score)

        rows.append(
            {
                "example_id": example.get("id", ""),
                "lang": lang,
                "subset": subset_name,
                "model_name": PRIMARY_MODEL_DISPLAY_NAME,
                "backend_name": BACKEND_NAME,
                "question": question,
                "prediction": prediction,
                "gold_answers": " | ".join(gold_answers),
                "exact_match": em if em is not None else 0,
                "f1": f1 if f1 is not None else 0.0,
                "wall_time_sec": result["wall_time_sec"],
                "total_duration_sec": result["total_duration_sec"],
                "load_duration_sec": result["load_duration_sec"],
                "prompt_eval_count": result["prompt_eval_count"],
                "eval_count": result["eval_count"],
                "prompt_tokens_per_sec": result["prompt_tokens_per_sec"],
                "generation_tokens_per_sec": result["generation_tokens_per_sec"],
                "mem_before_mb": round(mem_before, 2),
                "mem_after_mb": round(mem_after, 2),
            }
        )

    df = pd.DataFrame(rows)

    summary = {
        "model_name": PRIMARY_MODEL_DISPLAY_NAME,
        "backend_name": BACKEND_NAME,
        "lang": lang,
        "subset": subset_name,
        "n_examples": len(df),
        "skipped_no_answers": skipped_no_answers,
        "avg_em": round(float(df["exact_match"].mean()), 4) if len(df) else 0.0,
        "avg_f1": round(float(df["f1"].mean()), 4) if len(df) else 0.0,
        "avg_wall_time_sec": round(float(df["wall_time_sec"].mean()), 4) if len(df) else 0.0,
        "avg_total_duration_sec": round(float(df["total_duration_sec"].mean()), 4) if len(df) else 0.0,
        "avg_load_duration_sec": round(float(df["load_duration_sec"].mean()), 4) if len(df) else 0.0,
        "avg_prompt_tokens_per_sec": round(float(df["prompt_tokens_per_sec"].dropna().mean()), 4) if len(df) else 0.0,
        "avg_generation_tokens_per_sec": round(float(df["generation_tokens_per_sec"].dropna().mean()), 4) if len(df) else 0.0,
        "avg_mem_before_mb": round(float(df["mem_before_mb"].mean()), 2) if len(df) else 0.0,
        "avg_mem_after_mb": round(float(df["mem_after_mb"].mean()), 2) if len(df) else 0.0,
    }

    return df, summary


def main():
    ua_ds = load_ua_squad_validation_subset(UA_QA_SUBSET_SIZE)
    en_ds = load_en_squad_validation_subset(EN_QA_SUBSET_SIZE)

    ua_df, ua_summary = run_eval(ua_ds, lang="uk", subset_name="ua_squad_small")
    en_df, en_summary = run_eval(en_ds, lang="en", subset_name="squad_small")

    details = pd.concat([ua_df, en_df], ignore_index=True)
    summary_df = pd.DataFrame([ua_summary, en_summary])

    details_path = RESULTS_DIR / "pilot_qa_details.csv"
    summary_path = RESULTS_DIR / "pilot_qa_summary.csv"

    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== SUMMARY ===")
    print(summary_df)
    print(f"\nSaved details to: {details_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()