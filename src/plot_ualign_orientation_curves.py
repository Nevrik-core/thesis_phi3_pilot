from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


QUANTIZATION_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
QUANTIZATION_INDEX = {name: i for i, name in enumerate(QUANTIZATION_ORDER)}


def quantization_sort_key(value: str) -> int:
    return QUANTIZATION_INDEX.get(str(value), 999)


def load_orientation_summary(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required = {
        "quantization_name",
        "task",
        "lang",
        "ethics_unacceptable_rate",
        "mean_predicted_label_valid",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["quantization_name"] = df["quantization_name"].astype(str)
    df["task"] = df["task"].astype(str)
    df["lang"] = df["lang"].astype(str)

    df = df[df["quantization_name"].isin(QUANTIZATION_ORDER)].copy()
    df["_order"] = df["quantization_name"].map(quantization_sort_key)

    return df.sort_values(["task", "lang", "_order"]).drop(columns="_order")


def plot_ethics_strictness(df: pd.DataFrame, output_path: Path) -> None:
    ethics = df[df["task"] == "ethics"].copy()

    plt.figure(figsize=(10, 5))

    for lang in ["en", "uk"]:
        d = ethics[ethics["lang"] == lang].copy()
        d["_order"] = d["quantization_name"].map(quantization_sort_key)
        d = d.sort_values("_order")

        plt.plot(
            d["quantization_name"],
            d["ethics_unacceptable_rate"],
            marker="o",
            label=f"{lang.upper()} strictness",
        )

    plt.ylim(0, 1.0)
    plt.xlabel("Quantization mode")
    plt.ylabel("Unacceptable prediction rate")
    plt.title("UAlign ETHICS: strictness curve across quantization levels")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Language")

    plt.text(
        0.01,
        -0.16,
        "Lower = more permissive; higher = more strict",
        transform=plt.gca().transAxes,
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {output_path}")


def plot_social_valence(df: pd.DataFrame, output_path: Path) -> None:
    social = df[df["task"] == "social_chemistry"].copy()

    plt.figure(figsize=(10, 5))

    for lang in ["en", "uk"]:
        d = social[social["lang"] == lang].copy()
        d["_order"] = d["quantization_name"].map(quantization_sort_key)
        d = d.sort_values("_order")

        plt.plot(
            d["quantization_name"],
            d["mean_predicted_label_valid"],
            marker="o",
            label=f"{lang.upper()} valence",
        )

    plt.ylim(0, 2.0)
    plt.xlabel("Quantization mode")
    plt.ylabel("Mean predicted label")
    plt.title("UAlign Social Chemistry: bad-neutral-good valence curve")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Language")

    plt.text(
        0.01,
        -0.18,
        "0 = bad; 1 = expected/neutral; 2 = good",
        transform=plt.gca().transAxes,
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {output_path}")


def build_directional_drift_vs_q8(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for task in ["ethics", "social_chemistry"]:
        for lang in ["en", "uk"]:
            subset = df[(df["task"] == task) & (df["lang"] == lang)].copy()

            if subset.empty:
                continue

            q8 = subset[subset["quantization_name"] == "Q8_0"]

            if q8.empty:
                continue

            q8_row = q8.iloc[0]

            for _, row in subset.iterrows():
                quant = row["quantization_name"]

                if task == "ethics":
                    score = row["ethics_unacceptable_rate"]
                    ref = q8_row["ethics_unacceptable_rate"]
                    drift = score - ref
                    axis_name = "strictness_drift_vs_q8"
                    interpretation = "positive = stricter, negative = more permissive"

                else:
                    score = row["mean_predicted_label_valid"]
                    ref = q8_row["mean_predicted_label_valid"]
                    drift = score - ref
                    axis_name = "valence_drift_vs_q8"
                    interpretation = "positive = more positive, negative = more negative"

                rows.append(
                    {
                        "task": task,
                        "lang": lang,
                        "quantization_name": quant,
                        "score": round(float(score), 4),
                        "q8_reference_score": round(float(ref), 4),
                        "drift_vs_q8": round(float(drift), 4),
                        "axis_name": axis_name,
                        "interpretation": interpretation,
                    }
                )

    result = pd.DataFrame(rows)
    result["_order"] = result["quantization_name"].map(quantization_sort_key)

    return (
        result.sort_values(["task", "lang", "_order"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def plot_directional_drift(drift: pd.DataFrame, output_path: Path) -> None:
    plot_df = drift[drift["quantization_name"] != "Q8_0"].copy()
    plot_df["case"] = plot_df["task"] + "_" + plot_df["lang"]

    cases = [
        "ethics_en",
        "ethics_uk",
        "social_chemistry_en",
        "social_chemistry_uk",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, quant in zip(axes, ["Q6_K", "Q5_K_M", "Q4_K_M"]):
        d = plot_df[plot_df["quantization_name"] == quant].copy()
        d["case"] = pd.Categorical(d["case"], categories=cases, ordered=True)
        d = d.sort_values("case")

        ax.bar(d["case"].astype(str), d["drift_vs_q8"])
        ax.axhline(0, linewidth=1)

        ax.set_title(f"{quant} vs Q8_0")
        ax.set_xlabel("Task/language")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Directional drift vs Q8_0")
    fig.suptitle("UAlign directional drift across quantization levels")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot UAlign strictness/valence curves across Q8-Q4 quantization levels."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to ualign_orientation_summary_v4.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for figures and CSV.",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v4",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_orientation_summary(input_path)

    drift = build_directional_drift_vs_q8(df)

    scores_path = output_dir / f"ualign_orientation_scores_q8_to_q4_{args.version}.csv"
    drift_path = output_dir / f"ualign_directional_drift_vs_q8_{args.version}.csv"

    df.to_csv(scores_path, index=False, encoding="utf-8-sig")
    drift.to_csv(drift_path, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {scores_path}")
    print(f"[SAVED] {drift_path}")

    plot_ethics_strictness(
        df,
        output_dir / f"ualign_ethics_strictness_curve_q8_to_q4_{args.version}.png",
    )

    plot_social_valence(
        df,
        output_dir / f"ualign_social_valence_curve_q8_to_q4_{args.version}.png",
    )

    plot_directional_drift(
        drift,
        output_dir / f"ualign_directional_drift_vs_q8_{args.version}.png",
    )

    print("\n=== DIRECTIONAL DRIFT VS Q8_0 ===")
    print(drift.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()