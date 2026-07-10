from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SLOW_REASONING_MODES = {"think", "reasoning_only"}


def short_name(name: str) -> str:
    replacements = {
        " Q4_K_M": "",
        "NVIDIA-Nemotron-3-Nano-4B": "Nemotron",
        "MamayLM-Gemma-3-4B-IT-v1.0": "MamayLM",
        "Phi-4-mini-instruct": "Phi-4-mini",
        "Gemma-3-4B-it": "Gemma-3",
        "Qwen3-4B-Instruct-2507": "Qwen3-Instruct",
        "Qwen3-4B-Thinking-2507": "Qwen3-Thinking",
        "Hunyuan-4B-Instruct": "Hunyuan",
        "llm-jp-3-3.7b-instruct": "llm-jp",
    }

    out = str(name)

    for old, new in replacements.items():
        out = out.replace(old, new)

    return out


def savefig(output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def plot_uk_qa_belebele(scoreboard: pd.DataFrame, output_dir: Path) -> None:
    df = scoreboard[~scoreboard["reasoning_mode"].isin(SLOW_REASONING_MODES)].copy()
    df = df.sort_values("uk_quality_avg", ascending=True)

    y = np.arange(len(df))
    height = 0.35

    plt.figure(figsize=(10, 6))
    plt.barh(y - height / 2, df["qa_uk_f1"], height, label="QA UK F1")
    plt.barh(y + height / 2, df["belebele_uk_acc"], height, label="BELEBELE UK accuracy")
    plt.yticks(y, df["short_name"])
    plt.xlim(0, 1)
    plt.xlabel("Score")
    plt.title("Ukrainian QA and BELEBELE, non-reasoning models")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend()

    savefig(output_dir, "01_uk_qa_belebele_non_reasoning.png")


def plot_en_qa_belebele(scoreboard: pd.DataFrame, output_dir: Path) -> None:
    df = scoreboard[~scoreboard["reasoning_mode"].isin(SLOW_REASONING_MODES)].copy()
    df = df.sort_values("en_quality_avg", ascending=True)

    y = np.arange(len(df))
    height = 0.35

    plt.figure(figsize=(10, 6))
    plt.barh(y - height / 2, df["qa_en_f1"], height, label="QA EN F1")
    plt.barh(y + height / 2, df["belebele_en_acc"], height, label="BELEBELE EN accuracy")
    plt.yticks(y, df["short_name"])
    plt.xlim(0, 1)
    plt.xlabel("Score")
    plt.title("English QA and BELEBELE, non-reasoning models")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend()

    savefig(output_dir, "02_en_qa_belebele_non_reasoning.png")


def plot_ualign_language(scoreboard: pd.DataFrame, output_dir: Path) -> None:
    df = scoreboard[~scoreboard["reasoning_mode"].isin(SLOW_REASONING_MODES)].copy()
    df["ualign_uk_avg"] = df[["ethics_uk_acc", "social_uk_acc"]].mean(axis=1)
    df["ualign_en_avg"] = df[["ethics_en_acc", "social_en_acc"]].mean(axis=1)
    df = df.sort_values("ualign_uk_avg", ascending=True)

    y = np.arange(len(df))
    height = 0.35

    plt.figure(figsize=(10, 6))
    plt.barh(y - height / 2, df["ualign_uk_avg"], height, label="UAlign UK avg")
    plt.barh(y + height / 2, df["ualign_en_avg"], height, label="UAlign EN avg")
    plt.yticks(y, df["short_name"])
    plt.xlim(0, 1)
    plt.xlabel("Accuracy")
    plt.title("UAlign average accuracy by language, non-reasoning models")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend()

    savefig(output_dir, "03_ualign_language_non_reasoning.png")


def plot_language_gap(scoreboard: pd.DataFrame, output_dir: Path) -> None:
    df = scoreboard[~scoreboard["reasoning_mode"].isin(SLOW_REASONING_MODES)].copy()
    df = df.sort_values("en_minus_uk_quality_gap", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["short_name"], df["en_minus_uk_quality_gap"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("EN average quality - UK average quality")
    plt.title("Language gap, non-reasoning models")
    plt.grid(True, axis="x", alpha=0.3)

    savefig(output_dir, "04_en_minus_uk_gap_non_reasoning.png")


def plot_quality_vs_latency(scoreboard: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(9, 6))

    for _, row in scoreboard.iterrows():
        plt.scatter(row["avg_wall_time_sec"], row["avg_quality"], s=60)
        plt.annotate(
            row["short_name"],
            (row["avg_wall_time_sec"], row["avg_quality"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Average wall time per sample, sec")
    plt.ylabel("Average quality")
    plt.title("Quality vs latency, all stage-1 models")
    plt.grid(True, alpha=0.3)

    savefig(output_dir, "05_quality_vs_latency_all_models.png")


def plot_generated_tokens(scoreboard: pd.DataFrame, output_dir: Path) -> None:
    df = scoreboard.sort_values("avg_eval_count", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["short_name"], df["avg_eval_count"])
    plt.xlabel("Average generated tokens")
    plt.title("Generated token cost, all stage-1 models")
    plt.grid(True, axis="x", alpha=0.3)

    savefig(output_dir, "06_generated_tokens_all_models.png")


def plot_quality_vs_memory(scoreboard: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(9, 6))

    for _, row in scoreboard.iterrows():
        plt.scatter(row["avg_model_process_peak_rss_mb"], row["avg_quality"], s=60)
        plt.annotate(
            row["short_name"],
            (row["avg_model_process_peak_rss_mb"], row["avg_quality"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Average model process peak RSS, MB")
    plt.ylabel("Average quality")
    plt.title("Quality vs memory footprint, all stage-1 models")
    plt.grid(True, alpha=0.3)

    savefig(output_dir, "07_quality_vs_memory_all_models.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Stage 1 model-selection figures from model scoreboard."
    )

    parser.add_argument(
        "--scoreboard",
        type=str,
        required=True,
        help="Path to stage1_model_scoreboard_*.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <scoreboard parent>/figures",
    )

    args = parser.parse_args()

    scoreboard_path = Path(args.scoreboard)
    output_dir = Path(args.output_dir) if args.output_dir else scoreboard_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    scoreboard = pd.read_csv(scoreboard_path)
    scoreboard["short_name"] = scoreboard["display_name"].map(short_name)

    plot_uk_qa_belebele(scoreboard, output_dir)
    plot_en_qa_belebele(scoreboard, output_dir)
    plot_ualign_language(scoreboard, output_dir)
    plot_language_gap(scoreboard, output_dir)
    plot_quality_vs_latency(scoreboard, output_dir)
    plot_generated_tokens(scoreboard, output_dir)
    plot_quality_vs_memory(scoreboard, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
