from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


# -----------------------------------------------------------------------------
# Configuration fallback
# -----------------------------------------------------------------------------
try:
    from config import (  # type: ignore
        RESULTS_DIR as CONFIG_RESULTS_DIR,
        UA_QA_SUBSET_SIZE as CONFIG_UA_QA_SUBSET_SIZE,
        EN_QA_SUBSET_SIZE as CONFIG_EN_QA_SUBSET_SIZE,
        BELEBELE_SUBSET_SIZE as CONFIG_BELEBELE_SUBSET_SIZE,
        OLLAMA_GGUF_MODELS as CONFIG_OLLAMA_GGUF_MODELS,
        WEEK2_GGUF_MODEL_KEYS as CONFIG_WEEK2_GGUF_MODEL_KEYS,
    )
except Exception:
    CONFIG_RESULTS_DIR = None
    CONFIG_UA_QA_SUBSET_SIZE = 100
    CONFIG_EN_QA_SUBSET_SIZE = 100
    CONFIG_BELEBELE_SUBSET_SIZE = 100
    CONFIG_OLLAMA_GGUF_MODELS = {}
    CONFIG_WEEK2_GGUF_MODEL_KEYS = []


FALLBACK_MODELS = {
    "professorf_bf16": {
        "backend_name": "ollama_cpu_gguf_professorf_bf16",
        "quantization_name": "BF16",
        "source_repo": "professorf/Phi-4-mini-instruct-gguf",
        "model_role": "external_high_precision_reference",
    },
    "bartowski_q8_0": {
        "backend_name": "ollama_cpu_gguf_bartowski_q8_0",
        "quantization_name": "Q8_0",
        "source_repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "model_role": "main_same_runtime_curve",
    },
    "bartowski_q6_k": {
        "backend_name": "ollama_cpu_gguf_bartowski_q6_k",
        "quantization_name": "Q6_K",
        "source_repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "model_role": "main_same_runtime_curve",
    },
    "bartowski_q5_k_m": {
        "backend_name": "ollama_cpu_gguf_bartowski_q5_k_m",
        "quantization_name": "Q5_K_M",
        "source_repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "model_role": "main_same_runtime_curve",
    },
    "bartowski_q4_k_m": {
        "backend_name": "ollama_cpu_gguf_bartowski_q4_k_m",
        "quantization_name": "Q4_K_M",
        "source_repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "model_role": "main_same_runtime_curve",
    },
}

FALLBACK_ORDER = [
    "professorf_bf16",
    "bartowski_q8_0",
    "bartowski_q6_k",
    "bartowski_q5_k_m",
    "bartowski_q4_k_m",
]


MODELS: dict[str, dict[str, Any]] = dict(FALLBACK_MODELS)
for key, value in getattr(CONFIG_OLLAMA_GGUF_MODELS, "items", lambda: [])():
    if key in FALLBACK_MODELS or key in CONFIG_WEEK2_GGUF_MODEL_KEYS:
        merged = dict(FALLBACK_MODELS.get(key, {}))
        merged.update(value)
        MODELS[key] = merged

MODEL_ORDER = list(CONFIG_WEEK2_GGUF_MODEL_KEYS) if CONFIG_WEEK2_GGUF_MODEL_KEYS else FALLBACK_ORDER
MODEL_ORDER = [key for key in MODEL_ORDER if key in MODELS]

UA_QA_SUBSET_SIZE = int(CONFIG_UA_QA_SUBSET_SIZE)
EN_QA_SUBSET_SIZE = int(CONFIG_EN_QA_SUBSET_SIZE)
BELEBELE_SUBSET_SIZE = int(CONFIG_BELEBELE_SUBSET_SIZE) if int(CONFIG_BELEBELE_SUBSET_SIZE) != 20 else 100


def resolve_results_dir() -> Path:
    """Use config.RESULTS_DIR if it contains the needed files, otherwise cwd."""
    candidates: list[Path] = []
    if CONFIG_RESULTS_DIR is not None:
        candidates.append(Path(CONFIG_RESULTS_DIR))
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path(__file__).resolve().parent.parent / "results")

    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("pilot_*_summary_ollama_cpu_gguf_*_v*.csv")):
            return candidate

    # Default to config or cwd even if empty, so error messages include paths.
    return Path(CONFIG_RESULTS_DIR) if CONFIG_RESULTS_DIR is not None else Path.cwd()


RESULTS_DIR = resolve_results_dir()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def extract_version_number(path: Path) -> int:
    match = re.search(r"_v(\d+)\.csv$", path.name)
    return int(match.group(1)) if match else -1


def next_output_version(results_dir: Path, prefix: str) -> int:
    versions = [extract_version_number(p) for p in results_dir.glob(f"{prefix}_v*.csv")]
    versions = [v for v in versions if v >= 0]
    return max(versions) + 1 if versions else 1


def find_latest(pattern: str) -> Path:
    candidates = list(RESULTS_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files found for pattern: {RESULTS_DIR / pattern}")
    candidates.sort(key=extract_version_number)
    return candidates[-1]


def safe_float(row: pd.Series, column: str) -> float | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    return float(row[column])


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def pct_change(reference: float | None, current: float | None) -> float | None:
    """Positive value means reduction/improvement compared with reference."""
    if reference is None or current is None or reference == 0:
        return None
    return (reference - current) / reference * 100.0


def ratio_pct(current: float | None, reference: float | None) -> float | None:
    if reference is None or current is None or reference == 0:
        return None
    return current / reference * 100.0


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_week2_summaries() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_key in MODEL_ORDER:
        cfg = MODELS[model_key]
        backend = cfg["backend_name"]

        qa_pattern = f"pilot_qa_summary_{backend}_ua{UA_QA_SUBSET_SIZE}_en{EN_QA_SUBSET_SIZE}_v*.csv"
        belebele_pattern = f"pilot_belebele_summary_{backend}_{BELEBELE_SUBSET_SIZE}_v*.csv"

        for benchmark, pattern in [("QA", qa_pattern), ("BELEBELE", belebele_pattern)]:
            path = find_latest(pattern)
            df = pd.read_csv(path)

            for _, row in df.iterrows():
                out = row.to_dict()
                out["benchmark"] = benchmark
                out["model_key"] = model_key
                out["summary_file"] = path.name
                out["model_order"] = MODEL_ORDER.index(model_key)
                out["quality_metric"] = "F1" if benchmark == "QA" else "accuracy"
                out["quality_score"] = row.get("avg_f1") if benchmark == "QA" else row.get("accuracy")
                out["secondary_quality_metric"] = "EM" if benchmark == "QA" else "invalid_answer_rate"
                out["secondary_quality_score"] = row.get("avg_em") if benchmark == "QA" else row.get("invalid_answer_rate")
                rows.append(out)

    if not rows:
        raise RuntimeError("No Week 2 summary rows loaded.")

    result = pd.DataFrame(rows)
    result = result.sort_values(["benchmark", "model_order", "lang"]).reset_index(drop=True)
    return result


# -----------------------------------------------------------------------------
# Compact tables
# -----------------------------------------------------------------------------
def build_compact_table(long_df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    subset = long_df[long_df["benchmark"] == benchmark].copy()
    rows: list[dict[str, Any]] = []

    for model_key in MODEL_ORDER:
        mdf = subset[subset["model_key"] == model_key]
        if mdf.empty:
            continue

        uk = mdf[mdf["lang"] == "uk"]
        en = mdf[mdf["lang"] == "en"]
        uk_row = uk.iloc[0] if not uk.empty else pd.Series(dtype="object")
        en_row = en.iloc[0] if not en.empty else pd.Series(dtype="object")

        quality_name = "f1" if benchmark == "QA" else "accuracy"
        uk_quality = safe_float(uk_row, "avg_f1") if benchmark == "QA" else safe_float(uk_row, "accuracy")
        en_quality = safe_float(en_row, "avg_f1") if benchmark == "QA" else safe_float(en_row, "accuracy")

        row = {
            "model_key": model_key,
            "quantization": mdf.iloc[0].get("quantization_name"),
            "source_repo": mdf.iloc[0].get("source_repo"),
            "model_role": mdf.iloc[0].get("model_role"),
            "benchmark": benchmark,
            f"uk_{quality_name}": round_or_none(uk_quality, 4),
            f"en_{quality_name}": round_or_none(en_quality, 4),
            f"en_minus_uk_{quality_name}": round_or_none((en_quality - uk_quality) if uk_quality is not None and en_quality is not None else None, 4),
            "uk_avg_wall_time_sec": round_or_none(safe_float(uk_row, "avg_wall_time_sec"), 4),
            "en_avg_wall_time_sec": round_or_none(safe_float(en_row, "avg_wall_time_sec"), 4),
            "avg_model_peak_rss_mb": round_or_none(mean_or_none([
                safe_float(uk_row, "avg_model_process_peak_rss_mb"),
                safe_float(en_row, "avg_model_process_peak_rss_mb"),
            ]), 2),
            "max_model_peak_rss_mb": round_or_none(max([
                v for v in [
                    safe_float(uk_row, "max_model_process_peak_rss_mb"),
                    safe_float(en_row, "max_model_process_peak_rss_mb"),
                ] if v is not None
            ]), 2),
            "avg_generation_tokens_per_sec": round_or_none(mean_or_none([
                safe_float(uk_row, "avg_generation_tokens_per_sec"),
                safe_float(en_row, "avg_generation_tokens_per_sec"),
            ]), 4),
        }

        if benchmark == "QA":
            row.update({
                "uk_em": round_or_none(safe_float(uk_row, "avg_em"), 4),
                "en_em": round_or_none(safe_float(en_row, "avg_em"), 4),
                "en_minus_uk_em": round_or_none(
                    (safe_float(en_row, "avg_em") - safe_float(uk_row, "avg_em"))
                    if safe_float(en_row, "avg_em") is not None and safe_float(uk_row, "avg_em") is not None
                    else None,
                    4,
                ),
            })
        else:
            row.update({
                "uk_invalid_rate": round_or_none(safe_float(uk_row, "invalid_answer_rate"), 4),
                "en_invalid_rate": round_or_none(safe_float(en_row, "invalid_answer_rate"), 4),
            })

        rows.append(row)

    compact = pd.DataFrame(rows)

    # Add resource reduction compared with BF16 reference and Q8 same-runtime reference.
    if not compact.empty:
        bf16_rss = compact.loc[compact["model_key"] == "professorf_bf16", "avg_model_peak_rss_mb"]
        q8_rss = compact.loc[compact["model_key"] == "bartowski_q8_0", "avg_model_peak_rss_mb"]
        bf16_ref = float(bf16_rss.iloc[0]) if not bf16_rss.empty else None
        q8_ref = float(q8_rss.iloc[0]) if not q8_rss.empty else None

        compact["memory_reduction_vs_bf16_pct"] = compact["avg_model_peak_rss_mb"].apply(
            lambda x: round_or_none(pct_change(bf16_ref, float(x) if pd.notna(x) else None), 2)
        )
        compact["memory_reduction_vs_q8_pct"] = compact["avg_model_peak_rss_mb"].apply(
            lambda x: round_or_none(pct_change(q8_ref, float(x) if pd.notna(x) else None), 2)
        )

        # Quality retention compared with Q8 for Bartowski curve.
        quality_col_uk = "uk_f1" if benchmark == "QA" else "uk_accuracy"
        quality_col_en = "en_f1" if benchmark == "QA" else "en_accuracy"
        q8_uk = compact.loc[compact["model_key"] == "bartowski_q8_0", quality_col_uk]
        q8_en = compact.loc[compact["model_key"] == "bartowski_q8_0", quality_col_en]
        q8_uk_ref = float(q8_uk.iloc[0]) if not q8_uk.empty else None
        q8_en_ref = float(q8_en.iloc[0]) if not q8_en.empty else None

        compact["uk_quality_retained_vs_q8_pct"] = compact[quality_col_uk].apply(
            lambda x: round_or_none(ratio_pct(float(x) if pd.notna(x) else None, q8_uk_ref), 2)
        )
        compact["en_quality_retained_vs_q8_pct"] = compact[quality_col_en].apply(
            lambda x: round_or_none(ratio_pct(float(x) if pd.notna(x) else None, q8_en_ref), 2)
        )

    return compact


def build_language_gaps(qa_compact: pd.DataFrame, belebele_compact: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in qa_compact.iterrows():
        rows.append({
            "benchmark": "QA",
            "model_key": row["model_key"],
            "quantization": row["quantization"],
            "metric": "F1",
            "uk_score": row.get("uk_f1"),
            "en_score": row.get("en_f1"),
            "en_minus_uk": row.get("en_minus_uk_f1"),
        })
        rows.append({
            "benchmark": "QA",
            "model_key": row["model_key"],
            "quantization": row["quantization"],
            "metric": "EM",
            "uk_score": row.get("uk_em"),
            "en_score": row.get("en_em"),
            "en_minus_uk": row.get("en_minus_uk_em"),
        })

    for _, row in belebele_compact.iterrows():
        rows.append({
            "benchmark": "BELEBELE",
            "model_key": row["model_key"],
            "quantization": row["quantization"],
            "metric": "accuracy",
            "uk_score": row.get("uk_accuracy"),
            "en_score": row.get("en_accuracy"),
            "en_minus_uk": row.get("en_minus_uk_accuracy"),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------
def build_report(
    long_df: pd.DataFrame,
    qa_compact: pd.DataFrame,
    belebele_compact: pd.DataFrame,
    language_gaps: pd.DataFrame,
) -> str:
    q4_qa = qa_compact[qa_compact["model_key"] == "bartowski_q4_k_m"].iloc[0]
    bf16_qa = qa_compact[qa_compact["model_key"] == "professorf_bf16"].iloc[0]
    q8_qa = qa_compact[qa_compact["model_key"] == "bartowski_q8_0"].iloc[0]

    q4_bel = belebele_compact[belebele_compact["model_key"] == "bartowski_q4_k_m"].iloc[0]

    report_qa_cols = [
        "quantization",
        "uk_f1",
        "en_f1",
        "en_minus_uk_f1",
        "uk_em",
        "en_em",
        "avg_model_peak_rss_mb",
        "memory_reduction_vs_bf16_pct",
        "uk_quality_retained_vs_q8_pct",
        "en_quality_retained_vs_q8_pct",
    ]
    report_bel_cols = [
        "quantization",
        "uk_accuracy",
        "en_accuracy",
        "en_minus_uk_accuracy",
        "uk_invalid_rate",
        "en_invalid_rate",
        "avg_model_peak_rss_mb",
        "memory_reduction_vs_bf16_pct",
        "uk_quality_retained_vs_q8_pct",
        "en_quality_retained_vs_q8_pct",
    ]

    qa_table = qa_compact[report_qa_cols].copy()
    bel_table = belebele_compact[report_bel_cols].copy()

    return f"""# Week 2 GGUF Matrix Report Draft

## 1. Мета тижня

На другому тижні було розширено MVP benchmark pipeline для дипломного дослідження про вплив квантизації на якість і ресурсну ефективність локального запуску Phi-4-mini на українських текстових задачах.

Основне методологічне уточнення: попереднє порівняння `Transformers BF16 CPU` проти `Ollama Q4 CPU` було корисним як practical deployment baseline, але воно змішувало precision і runtime backend. Тому для чистішого експериментального ядра було сформовано GGUF/Ollama CPU-only матрицю.

## 2. Зафіксована матриця моделей

- `BF16` від professorf використано як external high-precision GGUF reference.
- `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M` від bartowski використано як основну same-runtime quantization curve.
- Усі запуски виконано в CPU-only режимі через Ollama з `num_gpu=0`.

## 3. Реалізовані компоненти

- Додано model registry у `config.py`.
- Додано перемикання активної моделі через `ACTIVE_OLLAMA_MODEL_KEY`.
- Реалізовано runner `run_week2_gguf_matrix.py`, який послідовно запускає всі моделі.
- Розширено QA та BELEBELE evaluation scripts метаданими про model source, quantization pipeline, imatrix і роль моделі.
- Додано resource monitoring: model process RSS, client process RSS, system memory peak, wall time, load time, prompt throughput, generation throughput.

## 4. QA results: UA-SQuAD 100 vs SQuAD 100

{df_to_markdown(qa_table)}

Ключове спостереження: на QA Q4_K_M зменшує model peak RSS з приблизно {bf16_qa['avg_model_peak_rss_mb']:.2f} MB до {q4_qa['avg_model_peak_rss_mb']:.2f} MB, тобто на {q4_qa['memory_reduction_vs_bf16_pct']:.2f}% відносно BF16 reference. Водночас якість на 100-sample subset залишається конкурентною щодо Q8_0: для української F1 retention становить {q4_qa['uk_quality_retained_vs_q8_pct']:.2f}%, для англійської — {q4_qa['en_quality_retained_vs_q8_pct']:.2f}%.

## 5. BELEBELE results: ukr_Cyrl 100 vs eng_Latn 100

{df_to_markdown(bel_table)}

Ключове спостереження: на BELEBELE Q4_K_M також суттєво зменшує memory footprint і не демонструє деградації на цьому 100-sample subset. Українська accuracy для Q4_K_M становить {q4_bel['uk_accuracy']:.4f}, англійська accuracy — {q4_bel['en_accuracy']:.4f}.

## 6. Language gap

{df_to_markdown(language_gaps)}

На обох benchmark-ах зберігається стабільний розрив між англійською та українською. Це підтримує актуальність основного research question: чи впливає квантизація на українські задачі інакше, ніж на англійські контрольні задачі.

## 7. Проблеми та рішення

1. Було виявлено, що вимірювання пам’яті тільки Python-процесу не відображає пам’ять самої Ollama-моделі. Рішення: додано ResourceMonitor, який відстежує Ollama/llama/runner процеси.
2. Було виявлено, що одночасно завантажені Ollama-моделі спотворюють peak RSS. Рішення: runner перед кожною моделлю виконує `ollama stop` для відомих моделей.
3. Було виявлено, що direct HF tag для деяких моделей, наприклад Q3, може давати 404. Рішення: Q3 винесено в optional future step через manual Modelfile.
4. BF16 взято з professorf, а основну quantization curve — з bartowski. Рішення: BF16 позначено як external reference, а не як частину Bartowski same-runtime curve.

## 8. План на тиждень 3

- Зібрати графіки quality/resource trade-off: F1/accuracy vs model peak RSS, wall time vs quantization level.
- Додати третій benchmark: UAlign або Eval-UA-tion subset.
- Перевірити результати на більшому subset або повторити частину запусків для стабільності.
- Підготувати перший варіант методологічного розділу дипломної роботи.
"""


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("\n=== WEEK 2 GGUF RESULT COMPARISON ===")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Model order: {MODEL_ORDER}")

    long_df = load_week2_summaries()
    qa_compact = build_compact_table(long_df, "QA")
    belebele_compact = build_compact_table(long_df, "BELEBELE")
    language_gaps = build_language_gaps(qa_compact, belebele_compact)

    output_prefix = "week2_gguf_matrix"
    version = next_output_version(RESULTS_DIR, output_prefix)
    version_label = f"v{version}"

    long_path = RESULTS_DIR / f"{output_prefix}_long_{version_label}.csv"
    qa_path = RESULTS_DIR / f"{output_prefix}_qa_compact_{version_label}.csv"
    belebele_path = RESULTS_DIR / f"{output_prefix}_belebele_compact_{version_label}.csv"
    gaps_path = RESULTS_DIR / f"{output_prefix}_language_gaps_{version_label}.csv"
    report_path = RESULTS_DIR / f"{output_prefix}_weekly_report_draft_{version_label}.md"

    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    qa_compact.to_csv(qa_path, index=False, encoding="utf-8-sig")
    belebele_compact.to_csv(belebele_path, index=False, encoding="utf-8-sig")
    language_gaps.to_csv(gaps_path, index=False, encoding="utf-8-sig")

    report = build_report(long_df, qa_compact, belebele_compact, language_gaps)
    report_path.write_text(report, encoding="utf-8")

    print("\n=== QA COMPACT ===")
    print(qa_compact.to_string(index=False))

    print("\n=== BELEBELE COMPACT ===")
    print(belebele_compact.to_string(index=False))

    print("\n=== LANGUAGE GAPS ===")
    print(language_gaps.to_string(index=False))

    print("\nSaved:")
    print(f"- {long_path}")
    print(f"- {qa_path}")
    print(f"- {belebele_path}")
    print(f"- {gaps_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
