from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def safe_mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and pd.notna(v)]

    if not clean:
        return None

    return round(sum(clean) / len(clean), 4)


def first_non_empty(series: pd.Series, default: str = "") -> str:
    if series.empty:
        return default

    values = series.dropna().astype(str)
    values = values[values.str.strip() != ""]

    if values.empty:
        return default

    return values.iloc[0]


def load_manifest(input_dir: Path, version: str) -> pd.DataFrame:
    path = input_dir / f"stage1_100_matrix_manifest_{version}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    return pd.read_csv(path)


def load_summary_files(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("pilot_*_summary_*.csv"))

    if not paths:
        raise FileNotFoundError(f"No pilot summary CSV files found in {input_dir}")

    frames = []

    print("\n[LOAD] Summary files:")

    for path in paths:
        print(f"- {path.name}")

        df = pd.read_csv(path)
        df["source_file"] = path.name

        if "pilot_qa_summary" in path.name:
            df["benchmark"] = "qa"
        elif "pilot_belebele_summary" in path.name:
            df["benchmark"] = "belebele"
        elif "pilot_ualign_summary" in path.name:
            df["benchmark"] = "ualign"
        else:
            df["benchmark"] = "unknown"

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "task" not in out.columns:
        out["task"] = ""

    if "lang" not in out.columns:
        out["lang"] = ""

    for col in [
        "avg_f1",
        "avg_em",
        "accuracy",
        "invalid_answer_rate",
        "avg_wall_time_sec",
        "avg_total_duration_sec",
        "avg_eval_count",
        "avg_prompt_eval_count",
        "avg_prompt_tokens_per_sec",
        "avg_generation_tokens_per_sec",
        "avg_model_process_peak_rss_mb",
        "max_model_process_peak_rss_mb",
        "thinking_rate",
        "thinking_fallback_rate",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def get_metric(
    df: pd.DataFrame,
    *,
    benchmark: str,
    lang: str,
    task: str | None,
    metric: str,
) -> float | None:
    d = df[
        (df["benchmark"] == benchmark)
        & (df["lang"].astype(str) == lang)
    ]

    if task is not None:
        d = d[d["task"].astype(str) == task]

    if d.empty or metric not in d.columns:
        return None

    value = d[metric].dropna()

    if value.empty:
        return None

    return round(float(value.iloc[0]), 4)


def get_runtime_mean(df: pd.DataFrame, metric: str) -> float | None:
    if metric not in df.columns:
        return None

    values = pd.to_numeric(df[metric], errors="coerce").dropna()

    if values.empty:
        return None

    return round(float(values.mean()), 4)


def get_runtime_max(df: pd.DataFrame, metric: str) -> float | None:
    if metric not in df.columns:
        return None

    values = pd.to_numeric(df[metric], errors="coerce").dropna()

    if values.empty:
        return None

    return round(float(values.max()), 4)


def build_scoreboard(summary: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, manifest_row in manifest.sort_values("run_order").iterrows():
        backend_name = manifest_row["backend_name"]

        df = summary[summary["backend_name"] == backend_name].copy()

        if df.empty:
            print(f"[WARN] No summary rows for backend: {backend_name}")
            continue

        qa_uk_f1 = get_metric(df, benchmark="qa", lang="uk", task=None, metric="avg_f1")
        qa_en_f1 = get_metric(df, benchmark="qa", lang="en", task=None, metric="avg_f1")

        qa_uk_em = get_metric(df, benchmark="qa", lang="uk", task=None, metric="avg_em")
        qa_en_em = get_metric(df, benchmark="qa", lang="en", task=None, metric="avg_em")

        belebele_uk_acc = get_metric(
            df,
            benchmark="belebele",
            lang="uk",
            task=None,
            metric="accuracy",
        )
        belebele_en_acc = get_metric(
            df,
            benchmark="belebele",
            lang="en",
            task=None,
            metric="accuracy",
        )

        ethics_uk_acc = get_metric(
            df,
            benchmark="ualign",
            lang="uk",
            task="ethics",
            metric="accuracy",
        )
        ethics_en_acc = get_metric(
            df,
            benchmark="ualign",
            lang="en",
            task="ethics",
            metric="accuracy",
        )

        social_uk_acc = get_metric(
            df,
            benchmark="ualign",
            lang="uk",
            task="social_chemistry",
            metric="accuracy",
        )
        social_en_acc = get_metric(
            df,
            benchmark="ualign",
            lang="en",
            task="social_chemistry",
            metric="accuracy",
        )

        quality_values = [
            qa_uk_f1,
            qa_en_f1,
            belebele_uk_acc,
            belebele_en_acc,
            ethics_uk_acc,
            ethics_en_acc,
            social_uk_acc,
            social_en_acc,
        ]

        uk_quality = safe_mean([qa_uk_f1, belebele_uk_acc, ethics_uk_acc, social_uk_acc])
        en_quality = safe_mean([qa_en_f1, belebele_en_acc, ethics_en_acc, social_en_acc])
        avg_quality = safe_mean(quality_values)

        ua_en_gap = (
            round(float(en_quality) - float(uk_quality), 4)
            if uk_quality is not None and en_quality is not None
            else None
        )

        row = {
            "run_order": manifest_row.get("run_order"),
            "model_key": manifest_row.get("model_key"),
            "display_name": manifest_row.get("display_name"),
            "backend_name": backend_name,
            "base_model_family": manifest_row.get("base_model_family", ""),
            "reasoning_mode": manifest_row.get("reasoning_mode", ""),
            "ollama_think": manifest_row.get("ollama_think", ""),
            "model_size_class": manifest_row.get("model_size_class", ""),
            "architecture_note": manifest_row.get("architecture_note", ""),
            "quantization_name": manifest_row.get("quantization_name", ""),
            "source_repo": manifest_row.get("source_repo", ""),

            "qa_uk_f1": qa_uk_f1,
            "qa_en_f1": qa_en_f1,
            "qa_uk_em": qa_uk_em,
            "qa_en_em": qa_en_em,

            "belebele_uk_acc": belebele_uk_acc,
            "belebele_en_acc": belebele_en_acc,

            "ethics_uk_acc": ethics_uk_acc,
            "ethics_en_acc": ethics_en_acc,
            "social_uk_acc": social_uk_acc,
            "social_en_acc": social_en_acc,

            "uk_quality_avg": uk_quality,
            "en_quality_avg": en_quality,
            "avg_quality": avg_quality,
            "en_minus_uk_quality_gap": ua_en_gap,

            "avg_wall_time_sec": get_runtime_mean(df, "avg_wall_time_sec"),
            "avg_total_duration_sec": get_runtime_mean(df, "avg_total_duration_sec"),
            "avg_eval_count": get_runtime_mean(df, "avg_eval_count"),
            "avg_prompt_tokens_per_sec": get_runtime_mean(df, "avg_prompt_tokens_per_sec"),
            "avg_generation_tokens_per_sec": get_runtime_mean(
                df,
                "avg_generation_tokens_per_sec",
            ),
            "avg_model_process_peak_rss_mb": get_runtime_mean(
                df,
                "avg_model_process_peak_rss_mb",
            ),
            "max_model_process_peak_rss_mb": get_runtime_max(
                df,
                "max_model_process_peak_rss_mb",
            ),
            "avg_thinking_rate": get_runtime_mean(df, "thinking_rate"),
            "avg_thinking_fallback_rate": get_runtime_mean(df, "thinking_fallback_rate"),
            "avg_invalid_answer_rate": get_runtime_mean(df, "invalid_answer_rate"),
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if "avg_quality" in out.columns:
        out["quality_rank"] = out["avg_quality"].rank(
            ascending=False,
            method="min",
        ).astype("Int64")

    if "uk_quality_avg" in out.columns:
        out["uk_quality_rank"] = out["uk_quality_avg"].rank(
            ascending=False,
            method="min",
        ).astype("Int64")

    if "en_quality_avg" in out.columns:
        out["en_quality_rank"] = out["en_quality_avg"].rank(
            ascending=False,
            method="min",
        ).astype("Int64")

    return out.sort_values(["quality_rank", "run_order"]).reset_index(drop=True)


def build_long_summary(summary: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "model_key",
        "display_name",
        "backend_name",
        "base_model_family",
        "reasoning_mode",
        "ollama_think",
        "model_size_class",
        "architecture_note",
        "quantization_name",
        "source_repo",
    ]

    meta = manifest[meta_cols].copy()

    out = summary.merge(meta, on="backend_name", how="left", suffixes=("", "_manifest"))

    selected_cols = [
        "model_key",
        "display_name",
        "backend_name",
        "base_model_family",
        "reasoning_mode",
        "ollama_think",
        "benchmark",
        "task",
        "lang",
        "quantization_name",
        "n_examples",
        "avg_f1",
        "avg_em",
        "accuracy",
        "invalid_answer_rate",
        "avg_wall_time_sec",
        "avg_total_duration_sec",
        "avg_eval_count",
        "avg_prompt_tokens_per_sec",
        "avg_generation_tokens_per_sec",
        "avg_model_process_peak_rss_mb",
        "max_model_process_peak_rss_mb",
        "thinking_rate",
        "thinking_fallback_rate",
        "source_file",
    ]

    selected_cols = [col for col in selected_cols if col in out.columns]

    return out[selected_cols].copy()


def write_markdown_report(
    output_dir: Path,
    version: str,
    scoreboard: pd.DataFrame,
) -> Path:
    path = output_dir / f"stage1_model_scoreboard_{version}.md"

    top_cols = [
        "quality_rank",
        "display_name",
        "reasoning_mode",
        "avg_quality",
        "uk_quality_avg",
        "en_quality_avg",
        "en_minus_uk_quality_gap",
        "avg_wall_time_sec",
        "avg_eval_count",
        "avg_model_process_peak_rss_mb",
        "avg_invalid_answer_rate",
    ]

    top_cols = [col for col in top_cols if col in scoreboard.columns]

    try:
        table = scoreboard[top_cols].to_markdown(index=False)
    except Exception:
        table = scoreboard[top_cols].to_string(index=False)

    text = f"""# Stage 1 model scoreboard: {version}

## Scope

This report aggregates QA, BELEBELE and UAlign 100-sample summary files into one model-level scoreboard.

## Ranking logic

`avg_quality` is the mean of:

- QA Ukrainian F1
- QA English F1
- BELEBELE Ukrainian accuracy
- BELEBELE English accuracy
- UAlign Ethics Ukrainian accuracy
- UAlign Ethics English accuracy
- UAlign Social Chemistry Ukrainian accuracy
- UAlign Social Chemistry English accuracy

`en_minus_uk_quality_gap` is:

English average quality - Ukrainian average quality.

Positive values mean English performed better on average.

## Scoreboard

{table}
"""

    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Stage 1 model-level summary from 100-sample benchmark outputs."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/stage1_q4_100_v3",
        help="Directory with Stage 1 result CSV files.",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="stage1_q4_100_v3",
        help="Version label used in output filenames.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <input-dir>/stage1_summary.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "stage1_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== BUILD STAGE 1 SUMMARY ===")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Version: {args.version}")

    manifest = load_manifest(input_dir, args.version)
    summary_raw = load_summary_files(input_dir)
    summary = normalize_summary(summary_raw)

    scoreboard = build_scoreboard(summary, manifest)
    long_summary = build_long_summary(summary, manifest)

    scoreboard_path = output_dir / f"stage1_model_scoreboard_{args.version}.csv"
    long_path = output_dir / f"stage1_benchmark_long_summary_{args.version}.csv"
    report_path = write_markdown_report(output_dir, args.version, scoreboard)

    scoreboard.to_csv(scoreboard_path, index=False, encoding="utf-8-sig")
    long_summary.to_csv(long_path, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {scoreboard_path}")
    print(f"[SAVED] {long_path}")
    print(f"[SAVED] {report_path}")

    quick_cols = [
        "quality_rank",
        "display_name",
        "reasoning_mode",
        "avg_quality",
        "uk_quality_avg",
        "en_quality_avg",
        "en_minus_uk_quality_gap",
        "avg_wall_time_sec",
        "avg_eval_count",
        "avg_invalid_answer_rate",
    ]
    quick_cols = [col for col in quick_cols if col in scoreboard.columns]

    print("\n=== QUICK SCOREBOARD ===")
    print(scoreboard[quick_cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()