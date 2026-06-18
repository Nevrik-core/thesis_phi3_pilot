from pathlib import Path
import re

import pandas as pd

from config import RESULTS_DIR, UA_QA_SUBSET_SIZE, EN_QA_SUBSET_SIZE


MASTER_BACKEND_NAME = "transformers_cpu_master"
Q4_BACKEND_NAME = "ollama_cpu_q4_k_m"


def extract_version_number(path: Path) -> int:
    """
    Extracts trailing version number from filenames like:
    pilot_qa_summary_ollama_cpu_q4_k_m_ua100_en100_v2.csv
    """
    match = re.search(r"_v(\d+)\.csv$", path.name)

    if not match:
        return -1

    return int(match.group(1))


def find_latest_summary(results_dir: Path, backend_name: str) -> Path:
    pattern = (
        f"pilot_qa_summary_{backend_name}_"
        f"ua{UA_QA_SUBSET_SIZE}_en{EN_QA_SUBSET_SIZE}_v*.csv"
    )

    candidates = list(results_dir.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"No QA summary files found for backend '{backend_name}' "
            f"with pattern: {pattern}"
        )

    candidates.sort(key=extract_version_number)

    return candidates[-1]


def next_output_version(results_dir: Path, prefix: str) -> int:
    existing_versions = []

    for path in results_dir.glob(f"{prefix}_v*.csv"):
        version = extract_version_number(path)

        if version >= 0:
            existing_versions.append(version)

    if not existing_versions:
        return 1

    return max(existing_versions) + 1


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator is None or denominator == 0:
        return None

    return numerator / denominator


def safe_pct(value: float | None) -> float | None:
    if value is None:
        return None

    return value * 100


def get_metric(row: pd.Series, candidates: list[str]) -> float | None:
    """
    Gets first available metric from row.
    Useful because master and Ollama summaries use slightly different names
    for throughput columns.
    """
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            return float(row[column])

    return None


def build_comparison(master_df: pd.DataFrame, q4_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for lang in ["uk", "en"]:
        master_match = master_df[master_df["lang"] == lang]
        q4_match = q4_df[q4_df["lang"] == lang]

        if master_match.empty:
            raise ValueError(f"Master summary does not contain lang='{lang}'")

        if q4_match.empty:
            raise ValueError(f"Q4 summary does not contain lang='{lang}'")

        master = master_match.iloc[0]
        q4 = q4_match.iloc[0]

        master_em = float(master["avg_em"])
        q4_em = float(q4["avg_em"])

        master_f1 = float(master["avg_f1"])
        q4_f1 = float(q4["avg_f1"])

        master_wall = float(master["avg_wall_time_sec"])
        q4_wall = float(q4["avg_wall_time_sec"])

        master_peak_rss = float(master["avg_model_process_peak_rss_mb"])
        q4_peak_rss = float(q4["avg_model_process_peak_rss_mb"])

        master_max_peak_rss = float(master["max_model_process_peak_rss_mb"])
        q4_max_peak_rss = float(q4["max_model_process_peak_rss_mb"])

        master_system_peak = float(master["avg_system_used_memory_peak_mb"])
        q4_system_peak = float(q4["avg_system_used_memory_peak_mb"])

        master_prompt_tokens = get_metric(
            master,
            ["avg_prompt_eval_count", "avg_prompt_token_count"],
        )
        q4_prompt_tokens = get_metric(
            q4,
            ["avg_prompt_eval_count", "avg_prompt_token_count"],
        )

        master_generation_tokens = get_metric(
            master,
            ["avg_eval_count", "avg_generated_token_count"],
        )
        q4_generation_tokens = get_metric(
            q4,
            ["avg_eval_count", "avg_generated_token_count"],
        )

        master_generation_tps = get_metric(
            master,
            [
                "avg_generation_tokens_per_sec",
                "avg_effective_generation_tokens_per_sec",
            ],
        )
        q4_generation_tps = get_metric(
            q4,
            [
                "avg_generation_tokens_per_sec",
                "avg_effective_generation_tokens_per_sec",
            ],
        )

        em_drop_abs = master_em - q4_em
        f1_drop_abs = master_f1 - q4_f1

        em_retained = safe_ratio(q4_em, master_em)
        f1_retained = safe_ratio(q4_f1, master_f1)

        wall_speedup = safe_ratio(master_wall, q4_wall)
        rss_reduction = safe_ratio(master_peak_rss - q4_peak_rss, master_peak_rss)
        max_rss_reduction = safe_ratio(
            master_max_peak_rss - q4_max_peak_rss,
            master_max_peak_rss,
        )
        system_memory_reduction = safe_ratio(
            master_system_peak - q4_system_peak,
            master_system_peak,
        )

        rows.append(
            {
                "lang": lang,

                "master_backend": master["backend_name"],
                "q4_backend": q4["backend_name"],

                "master_model": master["model_name"],
                "q4_model": q4["model_name"],

                "master_quantization": master["quantization_name"],
                "q4_quantization": q4["quantization_name"],

                "master_version": master["experiment_version"],
                "q4_version": q4["experiment_version"],

                "n_examples_master": int(master["n_examples"]),
                "n_examples_q4": int(q4["n_examples"]),

                "master_em": round(master_em, 4),
                "q4_em": round(q4_em, 4),
                "em_drop_abs": round(em_drop_abs, 4),
                "em_retained_pct": round(safe_pct(em_retained), 2)
                if em_retained is not None
                else None,

                "master_f1": round(master_f1, 4),
                "q4_f1": round(q4_f1, 4),
                "f1_drop_abs": round(f1_drop_abs, 4),
                "f1_retained_pct": round(safe_pct(f1_retained), 2)
                if f1_retained is not None
                else None,

                "master_avg_wall_time_sec": round(master_wall, 4),
                "q4_avg_wall_time_sec": round(q4_wall, 4),
                "wall_time_speedup_x": round(wall_speedup, 2)
                if wall_speedup is not None
                else None,

                "master_avg_model_peak_rss_mb": round(master_peak_rss, 2),
                "q4_avg_model_peak_rss_mb": round(q4_peak_rss, 2),
                "avg_model_peak_rss_reduction_mb": round(
                    master_peak_rss - q4_peak_rss,
                    2,
                ),
                "avg_model_peak_rss_reduction_pct": round(
                    safe_pct(rss_reduction),
                    2,
                )
                if rss_reduction is not None
                else None,

                "master_max_model_peak_rss_mb": round(master_max_peak_rss, 2),
                "q4_max_model_peak_rss_mb": round(q4_max_peak_rss, 2),
                "max_model_peak_rss_reduction_pct": round(
                    safe_pct(max_rss_reduction),
                    2,
                )
                if max_rss_reduction is not None
                else None,

                "master_avg_system_memory_peak_mb": round(master_system_peak, 2),
                "q4_avg_system_memory_peak_mb": round(q4_system_peak, 2),
                "avg_system_memory_reduction_pct": round(
                    safe_pct(system_memory_reduction),
                    2,
                )
                if system_memory_reduction is not None
                else None,

                "master_avg_prompt_tokens": round(master_prompt_tokens, 2)
                if master_prompt_tokens is not None
                else None,
                "q4_avg_prompt_tokens": round(q4_prompt_tokens, 2)
                if q4_prompt_tokens is not None
                else None,

                "master_avg_generation_tokens": round(master_generation_tokens, 2)
                if master_generation_tokens is not None
                else None,
                "q4_avg_generation_tokens": round(q4_generation_tokens, 2)
                if q4_generation_tokens is not None
                else None,

                "master_generation_tokens_per_sec": round(master_generation_tps, 4)
                if master_generation_tps is not None
                else None,
                "q4_generation_tokens_per_sec": round(q4_generation_tps, 4)
                if q4_generation_tps is not None
                else None,
            }
        )

    return pd.DataFrame(rows)


def build_language_gap_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    uk = comparison_df[comparison_df["lang"] == "uk"].iloc[0]
    en = comparison_df[comparison_df["lang"] == "en"].iloc[0]

    rows = [
        {
            "metric": "EM",
            "master_uk": uk["master_em"],
            "master_en": en["master_em"],
            "master_en_minus_uk": round(en["master_em"] - uk["master_em"], 4),
            "q4_uk": uk["q4_em"],
            "q4_en": en["q4_em"],
            "q4_en_minus_uk": round(en["q4_em"] - uk["q4_em"], 4),
            "gap_change_after_q4": round(
                (en["q4_em"] - uk["q4_em"])
                - (en["master_em"] - uk["master_em"]),
                4,
            ),
        },
        {
            "metric": "F1",
            "master_uk": uk["master_f1"],
            "master_en": en["master_f1"],
            "master_en_minus_uk": round(en["master_f1"] - uk["master_f1"], 4),
            "q4_uk": uk["q4_f1"],
            "q4_en": en["q4_f1"],
            "q4_en_minus_uk": round(en["q4_f1"] - uk["q4_f1"], 4),
            "gap_change_after_q4": round(
                (en["q4_f1"] - uk["q4_f1"])
                - (en["master_f1"] - uk["master_f1"]),
                4,
            ),
        },
    ]

    return pd.DataFrame(rows)


def main():
    master_path = find_latest_summary(RESULTS_DIR, MASTER_BACKEND_NAME)
    q4_path = find_latest_summary(RESULTS_DIR, Q4_BACKEND_NAME)

    print("\n=== QA COMPARISON INPUTS ===")
    print(f"Master summary: {master_path}")
    print(f"Q4 summary:     {q4_path}")

    master_df = pd.read_csv(master_path)
    q4_df = pd.read_csv(q4_path)

    comparison_df = build_comparison(master_df, q4_df)
    language_gap_df = build_language_gap_table(comparison_df)

    output_prefix = (
        f"qa_comparison_master_vs_q4_"
        f"ua{UA_QA_SUBSET_SIZE}_en{EN_QA_SUBSET_SIZE}"
    )

    version = next_output_version(RESULTS_DIR, output_prefix)

    comparison_path = RESULTS_DIR / f"{output_prefix}_v{version}.csv"
    language_gap_path = RESULTS_DIR / f"{output_prefix}_language_gap_v{version}.csv"

    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    language_gap_df.to_csv(language_gap_path, index=False, encoding="utf-8-sig")

    print("\n=== MASTER VS Q4 COMPARISON ===")
    print(comparison_df.to_string(index=False))

    print("\n=== LANGUAGE GAP TABLE ===")
    print(language_gap_df.to_string(index=False))

    print(f"\nSaved comparison to: {comparison_path}")
    print(f"Saved language gap to: {language_gap_path}")


if __name__ == "__main__":
    main()