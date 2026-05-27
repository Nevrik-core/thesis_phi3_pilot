# src/analyze_speed_tokens.py
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    from config import RESULTS_DIR as CONFIG_RESULTS_DIR
except Exception:
    CONFIG_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


QUANTIZATION_ORDER = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
QUANTIZATION_INDEX = {name: i for i, name in enumerate(QUANTIZATION_ORDER)}


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def quantization_sort_key(value: str) -> int:
    return QUANTIZATION_INDEX.get(str(value), 999)


def safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None

    return num / den


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None

    return round(float(value), digits)


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)

    return pd.to_numeric(df[column], errors="coerce")


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column

    return None


def benchmark_from_filename(path: Path) -> str:
    name = path.name.lower()

    if "ualign" in name:
        return "UAlign"

    if "belebele" in name:
        return "BELEBELE"

    if "qa" in name:
        return "QA"

    return "unknown"


def is_detail_file(path: Path) -> bool:
    return "_details_" in path.name


def is_summary_file(path: Path) -> bool:
    return "_summary_" in path.name


def sort_quantized(df: pd.DataFrame) -> pd.DataFrame:
    if "quantization_name" not in df.columns:
        return df

    return (
        df.assign(_quantization_order=df["quantization_name"].map(quantization_sort_key))
        .sort_values(["benchmark", "task", "lang", "_quantization_order"])
        .drop(columns=["_quantization_order"])
        .reset_index(drop=True)
    )


def df_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No data._"

    small = df.head(max_rows).copy()

    try:
        return small.to_markdown(index=False)
    except Exception:
        return small.to_string(index=False)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def find_latest_input_dir(results_dir: Path) -> Path:
    forced_output_dir = os.getenv("EXPERIMENT_OUTPUT_DIR")
    if forced_output_dir:
        candidate = Path(forced_output_dir)

        if candidate.exists() and list(candidate.glob("pilot_*_details_*_v*.csv")):
            return candidate

    version_dirs: list[tuple[int, Path]] = []

    if results_dir.exists():
        for path in results_dir.iterdir():
            if not path.is_dir():
                continue

            if not re.fullmatch(r"v\d+", path.name):
                continue

            if list(path.glob("pilot_*_details_*_v*.csv")):
                version_dirs.append((int(path.name[1:]), path))

    if version_dirs:
        version_dirs.sort(key=lambda item: item[0])
        return version_dirs[-1][1]

    if list(results_dir.glob("pilot_*_details_*_v*.csv")):
        return results_dir

    raise FileNotFoundError(
        f"No pilot detail CSV files found in {results_dir} or its version subdirectories."
    )


def load_detail_files(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("pilot_*_details_*_v*.csv"))

    if not paths:
        raise FileNotFoundError(f"No detail files found in {input_dir}")

    frames: list[pd.DataFrame] = []

    print("\n[LOAD] Detail files:")

    for path in paths:
        benchmark = benchmark_from_filename(path)

        print(f"- {path.name} ({benchmark})")

        df = pd.read_csv(path)
        df["source_file"] = path.name
        df["benchmark"] = benchmark

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return normalize_detail_df(out)


def normalize_detail_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "quantization_name" not in out.columns:
        raise ValueError("Missing required column: quantization_name")

    if "lang" not in out.columns:
        if "language" in out.columns:
            out["lang"] = out["language"]
        else:
            out["lang"] = "unknown"

    if "task" not in out.columns:
        out["task"] = out["benchmark"]

    out["benchmark"] = out["benchmark"].fillna("unknown").astype(str)
    out["task"] = out["task"].fillna(out["benchmark"]).astype(str)
    out["lang"] = out["lang"].fillna("unknown").astype(str)
    out["quantization_name"] = out["quantization_name"].fillna("unknown").astype(str)

    # For QA and BELEBELE detail files, task can be empty/NaN.
    # In that case benchmark name is the correct task-level fallback.
    out.loc[out["task"].isin(["nan", "None", ""]), "task"] = out["benchmark"]
    out.loc[out["lang"].isin(["nan", "None", ""]), "lang"] = "unknown"

    # Standardize core numeric runtime/token columns if present.
    runtime_columns = [
        "wall_time_sec",
        "total_duration_sec",
        "load_duration_sec",
        "prompt_eval_duration_sec",
        "eval_duration_sec",
        "prompt_eval_count",
        "eval_count",
        "prompt_tokens_per_sec",
        "generation_tokens_per_sec",
        "model_process_peak_rss_mb",
        "system_memory_used_peak_mb",
        "client_process_peak_rss_mb",
        "f1",
        "em",
        "is_correct",
        "is_valid_answer",
    ]

    for column in runtime_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    # Make token columns explicit.
    if "prompt_eval_count" in out.columns:
        out["prompt_tokens"] = out["prompt_eval_count"]
    else:
        out["prompt_tokens"] = np.nan

    if "eval_count" in out.columns:
        out["generated_tokens"] = out["eval_count"]
    else:
        out["generated_tokens"] = np.nan

    out["total_tokens"] = out["prompt_tokens"].fillna(0) + out["generated_tokens"].fillna(0)

    return out


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def aggregate_detail_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "benchmark",
        "task",
        "lang",
        "quantization_name",
    ]

    optional_meta_cols = [
        "backend_name",
        "primary_model_name",
        "model_name",
        "model_role",
        "source_repo",
        "model_source_repo",
        "artifact_family",
        "quantization_pipeline",
        "imatrix_used",
    ]

    rows: list[dict[str, Any]] = []

    for group_key, group in df.groupby(group_cols, dropna=False):
        benchmark, task, lang, quantization = group_key

        n = len(group)

        prompt_tokens = numeric_series(group, "prompt_tokens")
        generated_tokens = numeric_series(group, "generated_tokens")
        total_tokens = numeric_series(group, "total_tokens")

        wall_time = numeric_series(group, "wall_time_sec")
        total_duration = numeric_series(group, "total_duration_sec")
        load_duration = numeric_series(group, "load_duration_sec")
        prompt_eval_duration = numeric_series(group, "prompt_eval_duration_sec")
        eval_duration = numeric_series(group, "eval_duration_sec")

        prompt_tps = numeric_series(group, "prompt_tokens_per_sec")
        generation_tps = numeric_series(group, "generation_tokens_per_sec")

        model_rss = numeric_series(group, "model_process_peak_rss_mb")
        system_memory = numeric_series(group, "system_memory_used_peak_mb")
        client_rss = numeric_series(group, "client_process_peak_rss_mb")

        total_prompt_tokens = prompt_tokens.sum(skipna=True)
        total_generated_tokens = generated_tokens.sum(skipna=True)
        total_all_tokens = total_tokens.sum(skipna=True)

        sum_prompt_eval_duration = prompt_eval_duration.sum(skipna=True)
        sum_eval_duration = eval_duration.sum(skipna=True)
        sum_wall_time = wall_time.sum(skipna=True)
        sum_total_duration = total_duration.sum(skipna=True)

        aggregate_prompt_tps = safe_div(total_prompt_tokens, sum_prompt_eval_duration)
        aggregate_generation_tps = safe_div(total_generated_tokens, sum_eval_duration)
        aggregate_total_tokens_per_wall_sec = safe_div(total_all_tokens, sum_wall_time)
        aggregate_total_tokens_per_ollama_sec = safe_div(total_all_tokens, sum_total_duration)

        row: dict[str, Any] = {
            "benchmark": benchmark,
            "task": task,
            "lang": lang,
            "quantization_name": quantization,
            "n": n,

            "total_prompt_tokens": round_or_none(total_prompt_tokens, 2),
            "total_generated_tokens": round_or_none(total_generated_tokens, 2),
            "total_tokens": round_or_none(total_all_tokens, 2),

            "avg_prompt_tokens_per_sample": round_or_none(prompt_tokens.mean(skipna=True), 4),
            "avg_generated_tokens_per_sample": round_or_none(generated_tokens.mean(skipna=True), 4),
            "avg_total_tokens_per_sample": round_or_none(total_tokens.mean(skipna=True), 4),
            "median_total_tokens_per_sample": round_or_none(total_tokens.median(skipna=True), 4),
            "p95_total_tokens_per_sample": round_or_none(total_tokens.quantile(0.95), 4),

            "avg_wall_time_sec": round_or_none(wall_time.mean(skipna=True), 4),
            "median_wall_time_sec": round_or_none(wall_time.median(skipna=True), 4),
            "p95_wall_time_sec": round_or_none(wall_time.quantile(0.95), 4),
            "total_wall_time_sec": round_or_none(sum_wall_time, 4),

            "avg_ollama_total_duration_sec": round_or_none(total_duration.mean(skipna=True), 4),
            "avg_load_duration_sec": round_or_none(load_duration.mean(skipna=True), 4),
            "avg_prompt_eval_duration_sec": round_or_none(prompt_eval_duration.mean(skipna=True), 4),
            "avg_eval_duration_sec": round_or_none(eval_duration.mean(skipna=True), 4),

            "avg_prompt_tokens_per_sec": round_or_none(prompt_tps.mean(skipna=True), 4),
            "avg_generation_tokens_per_sec": round_or_none(generation_tps.mean(skipna=True), 4),
            "aggregate_prompt_tokens_per_sec": round_or_none(aggregate_prompt_tps, 4),
            "aggregate_generation_tokens_per_sec": round_or_none(aggregate_generation_tps, 4),
            "aggregate_total_tokens_per_wall_sec": round_or_none(
                aggregate_total_tokens_per_wall_sec,
                4,
            ),
            "aggregate_total_tokens_per_ollama_sec": round_or_none(
                aggregate_total_tokens_per_ollama_sec,
                4,
            ),

            "avg_model_process_peak_rss_mb": round_or_none(model_rss.mean(skipna=True), 2),
            "max_model_process_peak_rss_mb": round_or_none(model_rss.max(skipna=True), 2),
            "avg_system_memory_used_peak_mb": round_or_none(system_memory.mean(skipna=True), 2),
            "avg_client_process_peak_rss_mb": round_or_none(client_rss.mean(skipna=True), 2),
        }

        # Metadata.
        for column in optional_meta_cols:
            if column in group.columns:
                values = group[column].dropna().astype(str).unique()
                row[column] = values[0] if len(values) else ""

        # Quality metrics.
        if benchmark == "QA":
            f1 = numeric_series(group, "f1")
            em = numeric_series(group, "em")

            row["quality_metric"] = "F1"
            row["quality_score"] = round_or_none(f1.mean(skipna=True), 4)
            row["secondary_quality_metric"] = "EM"
            row["secondary_quality_score"] = round_or_none(em.mean(skipna=True), 4)
            row["avg_f1"] = round_or_none(f1.mean(skipna=True), 4)
            row["avg_em"] = round_or_none(em.mean(skipna=True), 4)

        elif "is_correct" in group.columns:
            is_correct = numeric_series(group, "is_correct")
            correct_count = is_correct.sum(skipna=True)

            row["quality_metric"] = "accuracy"
            row["quality_score"] = round_or_none(is_correct.mean(skipna=True), 4)
            row["secondary_quality_metric"] = "correct_count"
            row["secondary_quality_score"] = round_or_none(correct_count, 2)
            row["accuracy"] = round_or_none(is_correct.mean(skipna=True), 4)
            row["correct_count"] = round_or_none(correct_count, 2)

            row["tokens_per_correct_answer"] = round_or_none(
                safe_div(total_all_tokens, correct_count),
                4,
            )

        else:
            row["quality_metric"] = ""
            row["quality_score"] = None
            row["secondary_quality_metric"] = ""
            row["secondary_quality_score"] = None

        if row.get("quality_score") not in [None, 0]:
            row["avg_total_tokens_per_quality_point"] = round_or_none(
                safe_div(row["avg_total_tokens_per_sample"], row["quality_score"]),
                4,
            )
            row["wall_time_per_quality_point"] = round_or_none(
                safe_div(row["avg_wall_time_sec"], row["quality_score"]),
                4,
            )
        else:
            row["avg_total_tokens_per_quality_point"] = None
            row["wall_time_per_quality_point"] = None

        rows.append(row)

    out = pd.DataFrame(rows)
    out = sort_quantized(out)

    return out


def add_reference_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    metrics = [
        "quality_score",
        "avg_wall_time_sec",
        "avg_ollama_total_duration_sec",
        "avg_prompt_tokens_per_sample",
        "avg_generated_tokens_per_sample",
        "avg_total_tokens_per_sample",
        "avg_prompt_tokens_per_sec",
        "avg_generation_tokens_per_sec",
        "aggregate_prompt_tokens_per_sec",
        "aggregate_generation_tokens_per_sec",
        "avg_model_process_peak_rss_mb",
        "tokens_per_correct_answer",
        "avg_total_tokens_per_quality_point",
        "wall_time_per_quality_point",
    ]

    index_cols = ["benchmark", "task", "lang"]

    for reference in ["BF16", "Q8_0"]:
        ref = out[out["quantization_name"] == reference][index_cols + metrics].copy()

        if ref.empty:
            continue

        rename = {metric: f"{metric}_{reference}" for metric in metrics}
        ref = ref.rename(columns=rename)

        out = out.merge(ref, on=index_cols, how="left")

        for metric in metrics:
            current = pd.to_numeric(out[metric], errors="coerce")
            reference_values = pd.to_numeric(out[f"{metric}_{reference}"], errors="coerce")

            out[f"{metric}_delta_vs_{reference}"] = (current - reference_values).round(4)

            out[f"{metric}_ratio_vs_{reference}_pct"] = (
                current / reference_values * 100
            ).round(2)

    return out


def build_compact_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    speed_cols = [
        "benchmark",
        "task",
        "lang",
        "quantization_name",
        "n",
        "quality_metric",
        "quality_score",
        "avg_wall_time_sec",
        "avg_ollama_total_duration_sec",
        "avg_prompt_tokens_per_sample",
        "avg_generated_tokens_per_sample",
        "avg_total_tokens_per_sample",
        "avg_prompt_tokens_per_sec",
        "avg_generation_tokens_per_sec",
        "aggregate_prompt_tokens_per_sec",
        "aggregate_generation_tokens_per_sec",
        "avg_model_process_peak_rss_mb",
    ]

    efficiency_cols = [
        "benchmark",
        "task",
        "lang",
        "quantization_name",
        "quality_metric",
        "quality_score",
        "total_tokens",
        "avg_total_tokens_per_sample",
        "tokens_per_correct_answer",
        "avg_total_tokens_per_quality_point",
        "wall_time_per_quality_point",
        "quality_score_delta_vs_BF16",
        "quality_score_delta_vs_Q8_0",
        "avg_wall_time_sec_delta_vs_BF16",
        "avg_wall_time_sec_delta_vs_Q8_0",
        "avg_generation_tokens_per_sec_delta_vs_BF16",
        "avg_generation_tokens_per_sec_delta_vs_Q8_0",
        "avg_model_process_peak_rss_mb_delta_vs_BF16",
        "avg_model_process_peak_rss_mb_delta_vs_Q8_0",
    ]

    speed_cols = [col for col in speed_cols if col in summary.columns]
    efficiency_cols = [col for col in efficiency_cols if col in summary.columns]

    return summary[speed_cols].copy(), summary[efficiency_cols].copy()


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def ensure_figures_dir(output_dir: Path) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def savefig(figures_dir: Path, name: str) -> None:
    path = figures_dir / name

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")


def plot_metric_lines(
    summary: pd.DataFrame,
    figures_dir: Path,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    if metric not in summary.columns:
        return

    df = summary.copy()
    df["series"] = (
        df["benchmark"].fillna("unknown").astype(str)
        + "_"
        + df["task"].fillna("unknown").astype(str)
        + "_"
        + df["lang"].fillna("unknown").astype(str)
    )

    plt.figure(figsize=(11, 6))

    for series in sorted(df["series"].dropna().astype(str).unique()):
        d = df[df["series"] == series].copy()
        d["_order"] = d["quantization_name"].map(quantization_sort_key)
        d = d.sort_values("_order")

        plt.plot(
            d["quantization_name"],
            d[metric],
            marker="o",
            label=series,
        )

    plt.title(title)
    plt.xlabel("Quantization mode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Benchmark/task/lang", fontsize=8)

    savefig(figures_dir, filename)


def plot_wall_time_vs_generation_speed(summary: pd.DataFrame, figures_dir: Path) -> None:
    required = ["avg_wall_time_sec", "avg_generation_tokens_per_sec"]

    if any(col not in summary.columns for col in required):
        return

    df = summary.copy()

    plt.figure(figsize=(9, 6))

    for quantization in QUANTIZATION_ORDER:
        d = df[df["quantization_name"] == quantization]

        if d.empty:
            continue

        plt.scatter(
            d["avg_wall_time_sec"],
            d["avg_generation_tokens_per_sec"],
            label=quantization,
            s=60,
        )

    plt.title("Wall time vs generation throughput")
    plt.xlabel("Average wall time per sample, sec")
    plt.ylabel("Average generation tokens/sec")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Quantization")

    savefig(figures_dir, "04_wall_time_vs_generation_tokens_per_sec.png")


def plot_memory_vs_generation_speed(summary: pd.DataFrame, figures_dir: Path) -> None:
    required = ["avg_model_process_peak_rss_mb", "avg_generation_tokens_per_sec"]

    if any(col not in summary.columns for col in required):
        return

    df = summary.dropna(subset=required).copy()

    if df.empty:
        return

    plt.figure(figsize=(9, 6))

    for quantization in QUANTIZATION_ORDER:
        d = df[df["quantization_name"] == quantization]

        if d.empty:
            continue

        plt.scatter(
            d["avg_model_process_peak_rss_mb"],
            d["avg_generation_tokens_per_sec"],
            label=quantization,
            s=60,
        )

    plt.title("Memory footprint vs generation throughput")
    plt.xlabel("Average model peak RSS, MB")
    plt.ylabel("Average generation tokens/sec")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Quantization")

    savefig(figures_dir, "05_memory_vs_generation_tokens_per_sec.png")


def plot_quality_vs_total_tokens(summary: pd.DataFrame, figures_dir: Path) -> None:
    required = ["quality_score", "avg_total_tokens_per_sample"]

    if any(col not in summary.columns for col in required):
        return

    df = summary.dropna(subset=required).copy()

    if df.empty:
        return

    plt.figure(figsize=(9, 6))

    for quantization in QUANTIZATION_ORDER:
        d = df[df["quantization_name"] == quantization]

        if d.empty:
            continue

        plt.scatter(
            d["avg_total_tokens_per_sample"],
            d["quality_score"],
            label=quantization,
            s=60,
        )

    plt.title("Quality score vs average total tokens per sample")
    plt.xlabel("Average total tokens per sample")
    plt.ylabel("Quality score")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Quantization")

    savefig(figures_dir, "06_quality_vs_avg_total_tokens_per_sample.png")


def build_all_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = ensure_figures_dir(output_dir)

    plot_metric_lines(
        summary,
        figures_dir,
        metric="avg_generation_tokens_per_sec",
        title="Generation throughput by benchmark/task/language",
        ylabel="Average generation tokens/sec",
        filename="01_generation_tokens_per_sec.png",
    )

    plot_metric_lines(
        summary,
        figures_dir,
        metric="avg_prompt_tokens_per_sec",
        title="Prompt processing throughput by benchmark/task/language",
        ylabel="Average prompt tokens/sec",
        filename="02_prompt_tokens_per_sec.png",
    )

    plot_metric_lines(
        summary,
        figures_dir,
        metric="avg_total_tokens_per_sample",
        title="Average token cost per sample",
        ylabel="Average prompt + generated tokens",
        filename="03_avg_total_tokens_per_sample.png",
    )

    plot_wall_time_vs_generation_speed(summary, figures_dir)
    plot_memory_vs_generation_speed(summary, figures_dir)
    plot_quality_vs_total_tokens(summary, figures_dir)


# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------
def write_report(
    output_dir: Path,
    version_label: str,
    summary: pd.DataFrame,
    compact_speed: pd.DataFrame,
    compact_efficiency: pd.DataFrame,
) -> Path:
    report_path = output_dir / f"speed_token_report_{version_label}.md"

    report_cols = [
        "benchmark",
        "task",
        "lang",
        "quantization_name",
        "n",
        "quality_metric",
        "quality_score",
        "avg_wall_time_sec",
        "avg_prompt_tokens_per_sample",
        "avg_generated_tokens_per_sample",
        "avg_total_tokens_per_sample",
        "avg_prompt_tokens_per_sec",
        "avg_generation_tokens_per_sec",
        "avg_model_process_peak_rss_mb",
    ]

    report_cols = [col for col in report_cols if col in summary.columns]

    text = f"""# Speed and Token Efficiency Analysis {version_label}

## 1. Scope

This report separates practical latency from internal token throughput.

- `wall_time_sec` is interpreted as end-to-end latency per sample.
- `prompt_tokens_per_sec` measures prompt/context processing speed.
- `generation_tokens_per_sec` measures answer generation speed and is used as the closest available proxy for "cleaner" model generation speed in the Ollama/GGUF setup.
- token counts are used as an additional resource-cost characteristic.

## 2. Why this layer is needed

Memory usage alone is not enough to describe resource efficiency. A quantized model may use less RAM but not necessarily produce lower end-to-end latency. Therefore, latency, throughput, and token cost are analyzed separately.

## 3. Main compact summary

{df_to_markdown(summary[report_cols], max_rows=80)}

## 4. Suggested thesis wording

For speed evaluation, two groups of metrics were used. The first group describes practical user-facing latency: `wall_time_sec`, measured as end-to-end time of one request from the Python client to the local Ollama API and back. The second group describes internal inference throughput reported by Ollama: `prompt_tokens_per_sec` for prompt evaluation and `generation_tokens_per_sec` for response generation. This separation is necessary because wall time includes not only model computation, but also API/runtime overhead.

Token counts were additionally analyzed as a resource-cost characteristic. For each benchmark, the study reports average prompt tokens, generated tokens, total tokens per sample, and total token volume. This allows comparing quantization modes not only by accuracy and memory footprint, but also by how much token work was processed during evaluation.

## 5. Main output files

- `speed_token_summary_{version_label}.csv`
- `speed_token_summary_with_deltas_{version_label}.csv`
- `speed_token_compact_{version_label}.csv`
- `speed_token_efficiency_{version_label}.csv`
- `figures/01_generation_tokens_per_sec.png`
- `figures/02_prompt_tokens_per_sec.png`
- `figures/03_avg_total_tokens_per_sample.png`
- `figures/04_wall_time_vs_generation_tokens_per_sec.png`
- `figures/05_memory_vs_generation_tokens_per_sec.png`
- `figures/06_quality_vs_avg_total_tokens_per_sample.png`
"""

    report_path.write_text(text, encoding="utf-8")
    return report_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze speed, token throughput, and token cost from benchmark detail files."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory with pilot_*_details_* CSV files. Default: latest results/v* directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <input-dir>/speed_token_analysis.",
    )

    parser.add_argument(
        "--version",
        type=str,
        default=os.getenv("SPEED_TOKEN_ANALYSIS_VERSION", "v4"),
        help="Output version label. Default: v4.",
    )

    args = parser.parse_args()

    results_dir = Path(CONFIG_RESULTS_DIR)

    input_dir = Path(args.input_dir) if args.input_dir else find_latest_input_dir(results_dir)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_dir / "speed_token_analysis"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== SPEED AND TOKEN EFFICIENCY ANALYSIS ===")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Version: {args.version}")

    details = load_detail_files(input_dir)

    summary = aggregate_detail_metrics(details)
    summary_with_deltas = add_reference_deltas(summary)

    compact_speed, compact_efficiency = build_compact_tables(summary_with_deltas)

    version = args.version

    paths = {
        f"speed_token_summary_{version}.csv": summary,
        f"speed_token_summary_with_deltas_{version}.csv": summary_with_deltas,
        f"speed_token_compact_{version}.csv": compact_speed,
        f"speed_token_efficiency_{version}.csv": compact_efficiency,
    }

    for filename, table in paths.items():
        path = output_dir / filename
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {path}")

    build_all_plots(summary_with_deltas, output_dir)

    report_path = write_report(
        output_dir=output_dir,
        version_label=version,
        summary=summary_with_deltas,
        compact_speed=compact_speed,
        compact_efficiency=compact_efficiency,
    )

    print(f"[SAVED] {report_path}")

    print("\n=== QUICK SPEED/TOKEN SUMMARY ===")

    quick_cols = [
        "benchmark",
        "task",
        "lang",
        "quantization_name",
        "n",
        "quality_score",
        "avg_wall_time_sec",
        "avg_total_tokens_per_sample",
        "avg_prompt_tokens_per_sec",
        "avg_generation_tokens_per_sec",
        "avg_model_process_peak_rss_mb",
    ]

    quick_cols = [col for col in quick_cols if col in summary_with_deltas.columns]

    print(summary_with_deltas[quick_cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()