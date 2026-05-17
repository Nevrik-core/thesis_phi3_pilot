# src/build_week3_figures.py
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS_DIR


FIGURES_ROOT = RESULTS_DIR / "week3_figures"

QA_SUMMARY_PATTERN = "pilot_qa_summary_*_v*.csv"
BELEBELE_SUMMARY_PATTERN = "pilot_belebele_summary_*_v*.csv"
UALIGN_SUMMARY_PATTERN = "pilot_ualign_summary_*_v*.csv"

QUANTIZATION_ORDER = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]


def next_dir_version(root: Path) -> str:
    existing = []

    if not root.exists():
        return "v1"

    for path in root.iterdir():
        if not path.is_dir():
            continue

        name = path.name

        if name.startswith("v") and name[1:].isdigit():
            existing.append(int(name[1:]))

    if not existing:
        return "v1"

    return f"v{max(existing) + 1}"


def resolve_output_context() -> tuple[Path, Path, str]:
    forced_version = os.getenv("EXPERIMENT_VERSION")
    forced_output_dir = os.getenv("EXPERIMENT_OUTPUT_DIR")

    if forced_output_dir:
        search_root = Path(forced_output_dir)
        experiment_version = forced_version or search_root.name
        output_dir = search_root / "week3_figures"
    else:
        FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
        experiment_version = forced_version or next_dir_version(FIGURES_ROOT)
        search_root = RESULTS_DIR
        output_dir = FIGURES_ROOT / experiment_version

    output_dir.mkdir(parents=True, exist_ok=True)

    return search_root, output_dir, experiment_version


def read_csv_files(paths: list[Path]) -> pd.DataFrame:
    frames = []

    for path in paths:
        try:
            df = pd.read_csv(path)
            df["source_file"] = str(path)
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {repr(exc)}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "quantization_name" in out.columns:
        out["quantization_name"] = out["quantization_name"].astype(str)

    if "lang" in out.columns:
        out["lang"] = out["lang"].astype(str)

    if "task" in out.columns:
        out["task"] = out["task"].astype(str)

    if "avg_model_process_peak_rss_mb" in out.columns:
        out["avg_model_process_peak_rss_gb"] = (
            out["avg_model_process_peak_rss_mb"] / 1024.0
        )

    if "max_model_process_peak_rss_mb" in out.columns:
        out["max_model_process_peak_rss_gb"] = (
            out["max_model_process_peak_rss_mb"] / 1024.0
        )

    return out


def add_quantization_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "quantization_name" not in out.columns:
        return out

    order_map = {name: idx for idx, name in enumerate(QUANTIZATION_ORDER)}
    out["_quant_order"] = out["quantization_name"].map(order_map).fillna(999).astype(int)

    sort_cols = ["_quant_order", "quantization_name"]

    if "task" in out.columns:
        sort_cols.append("task")

    if "lang" in out.columns:
        sort_cols.append("lang")

    return out.sort_values(sort_cols)


def keep_latest_per_backend_lang_task(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "experiment_version" not in out.columns:
        return out

    out["_version_num"] = (
        out["experiment_version"]
        .astype(str)
        .str.extract(r"v(\d+)", expand=False)
        .fillna("0")
        .astype(int)
    )

    group_cols = [
        col
        for col in [
            "benchmark",
            "task",
            "backend_name",
            "quantization_name",
            "lang",
            "subset",
        ]
        if col in out.columns
    ]

    if not group_cols:
        return out.drop(columns=["_version_num"], errors="ignore")

    out = (
        out.sort_values("_version_num")
        .groupby(group_cols, as_index=False, dropna=False)
        .tail(1)
    )

    return out.drop(columns=["_version_num"], errors="ignore")


def pivot_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
):
    if df.empty:
        print(f"[SKIP] Empty dataframe for {output_path.name}")
        return

    required = {"quantization_name", "lang", metric}
    missing = required - set(df.columns)

    if missing:
        print(f"[SKIP] Missing columns {missing} for {output_path.name}")
        return

    plot_df = df[["quantization_name", "lang", metric]].dropna()
    plot_df = add_quantization_order(plot_df)

    if plot_df.empty:
        print(f"[SKIP] No data for {output_path.name}")
        return

    pivot = plot_df.pivot_table(
        index="quantization_name",
        columns="lang",
        values=metric,
        aggfunc="mean",
    )

    available_order = [q for q in QUANTIZATION_ORDER if q in pivot.index]
    remaining = [q for q in pivot.index if q not in available_order]
    pivot = pivot.loc[available_order + remaining]

    ax = pivot.plot(marker="o", figsize=(9, 5))

    ax.set_title(title)
    ax.set_xlabel("Quantization mode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"[SAVED] {output_path}")


def scatter_tradeoff(
    df: pd.DataFrame,
    quality_metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
):
    if df.empty:
        print(f"[SKIP] Empty dataframe for {output_path.name}")
        return

    required = {
        "quantization_name",
        "lang",
        quality_metric,
        "avg_model_process_peak_rss_gb",
    }
    missing = required - set(df.columns)

    if missing:
        print(f"[SKIP] Missing columns {missing} for {output_path.name}")
        return

    plot_df = df[
        [
            "quantization_name",
            "lang",
            quality_metric,
            "avg_model_process_peak_rss_gb",
        ]
    ].dropna()

    if plot_df.empty:
        print(f"[SKIP] No data for {output_path.name}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for lang, lang_df in plot_df.groupby("lang"):
        ax.scatter(
            lang_df["avg_model_process_peak_rss_gb"],
            lang_df[quality_metric],
            label=lang,
        )

        for _, row in lang_df.iterrows():
            ax.annotate(
                row["quantization_name"],
                (
                    row["avg_model_process_peak_rss_gb"],
                    row[quality_metric],
                ),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel("Average model peak RSS, GB")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Language / task")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"[SAVED] {output_path}")


def build_quality_resource_table(
    qa_df: pd.DataFrame,
    belebele_df: pd.DataFrame,
    ualign_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if not qa_df.empty:
        qa_cols = [
            "experiment_version",
            "backend_name",
            "model_name",
            "quantization_name",
            "lang",
            "n_examples",
            "avg_em",
            "avg_f1",
            "avg_wall_time_sec",
            "avg_prompt_eval_count",
            "avg_eval_count",
            "avg_prompt_tokens_per_sec",
            "avg_generation_tokens_per_sec",
            "avg_model_process_peak_rss_mb",
            "avg_model_process_peak_rss_gb",
            "max_model_process_peak_rss_mb",
        ]
        qa_cols = [col for col in qa_cols if col in qa_df.columns]
        temp = qa_df[qa_cols].copy()
        temp.insert(0, "benchmark", "qa_squad")
        temp.insert(1, "task", "extractive_qa")
        rows.append(temp)

    if not belebele_df.empty:
        belebele_cols = [
            "experiment_version",
            "backend_name",
            "model_name",
            "quantization_name",
            "lang",
            "n_examples",
            "accuracy",
            "invalid_answer_rate",
            "avg_wall_time_sec",
            "avg_prompt_tokens_per_sec",
            "avg_generation_tokens_per_sec",
            "avg_model_process_peak_rss_mb",
            "avg_model_process_peak_rss_gb",
            "max_model_process_peak_rss_mb",
        ]
        belebele_cols = [col for col in belebele_cols if col in belebele_df.columns]
        temp = belebele_df[belebele_cols].copy()
        temp.insert(0, "benchmark", "belebele")
        temp.insert(1, "task", "reading_comprehension_mc")
        rows.append(temp)

    if not ualign_df.empty:
        ualign_cols = [
            "experiment_version",
            "benchmark",
            "task",
            "backend_name",
            "model_name",
            "quantization_name",
            "lang",
            "n_examples",
            "accuracy",
            "invalid_answer_rate",
            "avg_wall_time_sec",
            "avg_prompt_eval_count",
            "avg_eval_count",
            "avg_prompt_tokens_per_sec",
            "avg_generation_tokens_per_sec",
            "avg_model_process_peak_rss_mb",
            "avg_model_process_peak_rss_gb",
            "max_model_process_peak_rss_mb",
        ]
        ualign_cols = [col for col in ualign_cols if col in ualign_df.columns]
        temp = ualign_df[ualign_cols].copy()
        rows.append(temp)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = add_quantization_order(combined)

    return combined.drop(columns=["_quant_order"], errors="ignore")


def find_summary_paths(search_root: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in search_root.rglob(pattern)
        if "week3_figures" not in str(path)
        and "qa_error_analysis" not in str(path)
    )


def main():
    search_root, output_dir, experiment_version = resolve_output_context()

    print("\n=== BUILD WEEK 3 FIGURES ===")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Search root: {search_root}")
    print(f"Output dir: {output_dir}")
    print(f"Figures version: {experiment_version}")

    qa_paths = find_summary_paths(search_root, QA_SUMMARY_PATTERN)
    belebele_paths = find_summary_paths(search_root, BELEBELE_SUMMARY_PATTERN)
    ualign_paths = find_summary_paths(search_root, UALIGN_SUMMARY_PATTERN)

    print(f"\nFound QA summary files: {len(qa_paths)}")
    for path in qa_paths:
        print(f"  - {path}")

    print(f"\nFound BELEBELE summary files: {len(belebele_paths)}")
    for path in belebele_paths:
        print(f"  - {path}")

    print(f"\nFound UAlign summary files: {len(ualign_paths)}")
    for path in ualign_paths:
        print(f"  - {path}")

    qa_df = normalize_summary(read_csv_files(qa_paths))
    belebele_df = normalize_summary(read_csv_files(belebele_paths))
    ualign_df = normalize_summary(read_csv_files(ualign_paths))

    qa_df = keep_latest_per_backend_lang_task(qa_df)
    belebele_df = keep_latest_per_backend_lang_task(belebele_df)
    ualign_df = keep_latest_per_backend_lang_task(ualign_df)

    qa_df = add_quantization_order(qa_df).drop(columns=["_quant_order"], errors="ignore")
    belebele_df = add_quantization_order(belebele_df).drop(
        columns=["_quant_order"],
        errors="ignore",
    )
    ualign_df = add_quantization_order(ualign_df).drop(
        columns=["_quant_order"],
        errors="ignore",
    )

    qa_combined_path = output_dir / f"week3_qa_summary_combined_{experiment_version}.csv"
    belebele_combined_path = (
        output_dir / f"week3_belebele_summary_combined_{experiment_version}.csv"
    )
    ualign_combined_path = (
        output_dir / f"week3_ualign_summary_combined_{experiment_version}.csv"
    )
    quality_resource_path = (
        output_dir / f"week3_quality_resource_table_{experiment_version}.csv"
    )

    qa_df.to_csv(qa_combined_path, index=False, encoding="utf-8-sig")
    belebele_df.to_csv(belebele_combined_path, index=False, encoding="utf-8-sig")
    ualign_df.to_csv(ualign_combined_path, index=False, encoding="utf-8-sig")

    quality_resource_df = build_quality_resource_table(qa_df, belebele_df, ualign_df)
    quality_resource_df.to_csv(
        quality_resource_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n[SAVED] {qa_combined_path}")
    print(f"[SAVED] {belebele_combined_path}")
    print(f"[SAVED] {ualign_combined_path}")
    print(f"[SAVED] {quality_resource_path}")

    pivot_metric(
        qa_df,
        metric="avg_f1",
        title="QA: UA-SQuAD vs SQuAD, F1",
        ylabel="F1",
        output_path=output_dir / f"qa_f1_by_quantization_{experiment_version}.png",
    )

    pivot_metric(
        belebele_df,
        metric="accuracy",
        title="BELEBELE: ukr_Cyrl vs eng_Latn, accuracy",
        ylabel="Accuracy",
        output_path=output_dir
        / f"belebele_accuracy_by_quantization_{experiment_version}.png",
    )

    if not ualign_df.empty:
        ualign_plot_df = ualign_df.copy()

        if "task" in ualign_plot_df.columns:
            ualign_plot_df["lang"] = (
                ualign_plot_df["lang"].astype(str)
                + "_"
                + ualign_plot_df["task"].astype(str)
            )

        pivot_metric(
            ualign_plot_df,
            metric="accuracy",
            title="UAlign: accuracy by language and task",
            ylabel="Accuracy",
            output_path=output_dir
            / f"ualign_accuracy_by_quantization_{experiment_version}.png",
        )

    peak_rows = []

    if not qa_df.empty and "avg_model_process_peak_rss_gb" in qa_df.columns:
        qa_peak = qa_df[
            ["quantization_name", "avg_model_process_peak_rss_gb"]
        ].copy()
        qa_peak["benchmark"] = "QA"
        peak_rows.append(qa_peak)

    if not belebele_df.empty and "avg_model_process_peak_rss_gb" in belebele_df.columns:
        belebele_peak = belebele_df[
            ["quantization_name", "avg_model_process_peak_rss_gb"]
        ].copy()
        belebele_peak["benchmark"] = "BELEBELE"
        peak_rows.append(belebele_peak)

    if not ualign_df.empty and "avg_model_process_peak_rss_gb" in ualign_df.columns:
        ualign_peak = ualign_df[
            ["quantization_name", "avg_model_process_peak_rss_gb"]
        ].copy()
        ualign_peak["benchmark"] = "UALIGN"
        peak_rows.append(ualign_peak)

    if peak_rows:
        peak_df = pd.concat(peak_rows, ignore_index=True)
        peak_df = (
            peak_df.groupby(["quantization_name", "benchmark"], as_index=False)
            .mean(numeric_only=True)
        )
        peak_df = peak_df.rename(columns={"benchmark": "lang"})

        pivot_metric(
            peak_df,
            metric="avg_model_process_peak_rss_gb",
            title="Average model peak RSS by quantization mode",
            ylabel="Average model peak RSS, GB",
            output_path=output_dir
            / f"peak_rss_by_quantization_{experiment_version}.png",
        )

    pivot_metric(
        qa_df,
        metric="avg_wall_time_sec",
        title="QA: average wall time per example",
        ylabel="Seconds per example",
        output_path=output_dir
        / f"qa_wall_time_by_quantization_{experiment_version}.png",
    )

    pivot_metric(
        belebele_df,
        metric="avg_wall_time_sec",
        title="BELEBELE: average wall time per example",
        ylabel="Seconds per example",
        output_path=output_dir
        / f"belebele_wall_time_by_quantization_{experiment_version}.png",
    )

    if not ualign_df.empty:
        ualign_wall_df = ualign_df.copy()

        if "task" in ualign_wall_df.columns:
            ualign_wall_df["lang"] = (
                ualign_wall_df["lang"].astype(str)
                + "_"
                + ualign_wall_df["task"].astype(str)
            )

        pivot_metric(
            ualign_wall_df,
            metric="avg_wall_time_sec",
            title="UAlign: average wall time per example",
            ylabel="Seconds per example",
            output_path=output_dir
            / f"ualign_wall_time_by_quantization_{experiment_version}.png",
        )

    pivot_metric(
        qa_df,
        metric="avg_generation_tokens_per_sec",
        title="QA: generation throughput",
        ylabel="Generation tokens/sec",
        output_path=output_dir
        / f"qa_generation_throughput_by_quantization_{experiment_version}.png",
    )

    pivot_metric(
        belebele_df,
        metric="avg_generation_tokens_per_sec",
        title="BELEBELE: generation throughput",
        ylabel="Generation tokens/sec",
        output_path=output_dir
        / f"belebele_generation_throughput_by_quantization_{experiment_version}.png",
    )

    if not ualign_df.empty:
        ualign_tps_df = ualign_df.copy()

        if "task" in ualign_tps_df.columns:
            ualign_tps_df["lang"] = (
                ualign_tps_df["lang"].astype(str)
                + "_"
                + ualign_tps_df["task"].astype(str)
            )

        pivot_metric(
            ualign_tps_df,
            metric="avg_generation_tokens_per_sec",
            title="UAlign: generation throughput",
            ylabel="Generation tokens/sec",
            output_path=output_dir
            / f"ualign_generation_throughput_by_quantization_{experiment_version}.png",
        )

    scatter_tradeoff(
        qa_df,
        quality_metric="avg_f1",
        title="QA quality-resource trade-off",
        ylabel="F1",
        output_path=output_dir
        / f"qa_tradeoff_f1_vs_peak_rss_{experiment_version}.png",
    )

    scatter_tradeoff(
        belebele_df,
        quality_metric="accuracy",
        title="BELEBELE quality-resource trade-off",
        ylabel="Accuracy",
        output_path=output_dir
        / f"belebele_tradeoff_accuracy_vs_peak_rss_{experiment_version}.png",
    )

    if not ualign_df.empty:
        ualign_tradeoff_df = ualign_df.copy()

        if "task" in ualign_tradeoff_df.columns:
            ualign_tradeoff_df["lang"] = (
                ualign_tradeoff_df["lang"].astype(str)
                + "_"
                + ualign_tradeoff_df["task"].astype(str)
            )

        scatter_tradeoff(
            ualign_tradeoff_df,
            quality_metric="accuracy",
            title="UAlign quality-resource trade-off",
            ylabel="Accuracy",
            output_path=output_dir
            / f"ualign_tradeoff_accuracy_vs_peak_rss_{experiment_version}.png",
        )

    print("\n=== WEEK 3 FIGURES BUILD COMPLETE ===")


if __name__ == "__main__":
    main()