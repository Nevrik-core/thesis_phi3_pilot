from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUANT_ORDER = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
QUANT_INDEX = {q: i for i, q in enumerate(QUANT_ORDER)}


def sort_quant(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(_order=df["quantization_name"].map(QUANT_INDEX))
        .sort_values("_order")
        .drop(columns="_order")
    )


def savefig(output_dir: Path, name: str) -> None:
    path = output_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def plot_ethics_unacceptable_rate(orientation: pd.DataFrame, output_dir: Path) -> None:
    ethics = orientation[orientation["task"] == "ethics"].copy()

    plt.figure(figsize=(8, 5))

    for lang in ["en", "uk"]:
        d = sort_quant(ethics[ethics["lang"] == lang])
        plt.plot(
            d["quantization_name"],
            d["ethics_unacceptable_rate"],
            marker="o",
            label=lang,
        )

    plt.title("UAlign ETHICS: rate of 'unacceptable' predictions")
    plt.xlabel("Quantization mode")
    plt.ylabel("Unacceptable prediction rate")
    plt.ylim(0, 1.05)
    plt.legend(title="Language")
    plt.grid(True, alpha=0.3)

    savefig(output_dir, "01_ethics_unacceptable_rate.png")


def plot_accuracy_by_task_lang(orientation: pd.DataFrame, output_dir: Path) -> None:
    df = orientation.copy()
    df["series"] = df["task"] + "_" + df["lang"]

    plt.figure(figsize=(10, 5))

    for series in sorted(df["series"].unique()):
        d = sort_quant(df[df["series"] == series])
        plt.plot(
            d["quantization_name"],
            d["accuracy"],
            marker="o",
            label=series,
        )

    plt.title("UAlign accuracy by task and language")
    plt.xlabel("Quantization mode")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend(title="Task/lang")
    plt.grid(True, alpha=0.3)

    savefig(output_dir, "02_ualign_accuracy_by_task_lang.png")


def plot_social_label_distribution(orientation: pd.DataFrame, output_dir: Path) -> None:
    for lang in ["en", "uk"]:
        d = sort_quant(
            orientation[
                (orientation["task"] == "social_chemistry")
                & (orientation["lang"] == lang)
            ]
        ).copy()

        x = np.arange(len(d))
        bottom = np.zeros(len(d))

        plt.figure(figsize=(8, 5))

        for col, label in [
            ("social_bad_rate", "bad"),
            ("social_neutral_rate", "expected/neutral"),
            ("social_good_rate", "good"),
        ]:
            values = d[col].fillna(0).to_numpy()
            plt.bar(x, values, bottom=bottom, label=label)
            bottom += values

        plt.title(f"UAlign Social Chemistry ({lang}): predicted label distribution")
        plt.xlabel("Quantization mode")
        plt.ylabel("Share of valid predictions")
        plt.xticks(x, d["quantization_name"])
        plt.ylim(0, 1.05)
        plt.legend(title="Predicted label")
        plt.grid(True, axis="y", alpha=0.3)

        savefig(output_dir, f"03_social_chemistry_{lang}_label_distribution.png")


def plot_flip_rates(flips: pd.DataFrame, output_dir: Path) -> None:
    for ref, target, filename_idx in [
        ("BF16", "Q4_K_M", "04"),
        ("Q8_0", "Q4_K_M", "05"),
    ]:
        d = flips[
            (flips["reference_quantization"] == ref)
            & (flips["target_quantization"] == target)
        ].copy()

        d["case"] = d["task"] + "_" + d["lang"]
        d = d.sort_values(["task", "lang"])

        plt.figure(figsize=(9, 5))

        x = np.arange(len(d))
        plt.bar(x, d["label_changed_rate"])

        plt.title(f"UAlign label flip rate: {ref} → {target}")
        plt.xlabel("Task/language")
        plt.ylabel("Label changed rate")
        plt.xticks(x, d["case"], rotation=20, ha="right")
        plt.ylim(0, max(0.25, float(d["label_changed_rate"].max()) + 0.05))
        plt.grid(True, axis="y", alpha=0.3)

        for i, row in enumerate(d.itertuples(index=False)):
            plt.text(
                i,
                row.label_changed_rate + 0.01,
                f"Δacc={row.accuracy_delta:+.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        savefig(output_dir, f"{filename_idx}_label_flip_rate_{ref}_to_{target}.png")


def plot_en_uk_consistency(consistency: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))

    for task in sorted(consistency["task"].unique()):
        d = sort_quant(consistency[consistency["task"] == task])
        plt.plot(
            d["quantization_name"],
            d["label_match_rate_on_valid_pairs"],
            marker="o",
            label=task,
        )

    plt.title("UAlign EN-UA consistency: same label on parallel examples")
    plt.xlabel("Quantization mode")
    plt.ylabel("EN-UA label match rate")
    plt.ylim(0, 1.05)
    plt.legend(title="Task")
    plt.grid(True, alpha=0.3)

    savefig(output_dir, "06_en_uk_label_consistency.png")


def plot_mean_predicted_label_drift(orientation: pd.DataFrame, output_dir: Path) -> None:
    df = orientation.copy()
    df["series"] = df["task"] + "_" + df["lang"]

    plt.figure(figsize=(10, 5))

    for series in sorted(df["series"].unique()):
        d = sort_quant(df[df["series"] == series])
        plt.plot(
            d["quantization_name"],
            d["mean_predicted_label_valid_delta_vs_BF16"],
            marker="o",
            label=series,
        )

    plt.axhline(0, linewidth=1)
    plt.title("UAlign behavior drift vs BF16: mean predicted label")
    plt.xlabel("Quantization mode")
    plt.ylabel("Mean predicted label delta vs BF16")
    plt.legend(title="Task/lang")
    plt.grid(True, alpha=0.3)

    savefig(output_dir, "07_mean_predicted_label_drift_vs_bf16.png")


def plot_confusion_matrices(confusion: pd.DataFrame, output_dir: Path) -> None:
    selected_quants = ["BF16", "Q8_0", "Q4_K_M"]

    for task in sorted(confusion["task"].unique()):
        for lang in sorted(confusion["lang"].unique()):
            for quant in selected_quants:
                d = confusion[
                    (confusion["task"] == task)
                    & (confusion["lang"] == lang)
                    & (confusion["quantization_name"] == quant)
                ].copy()

                if d.empty:
                    continue

                gold_labels = sorted(d["gold_label"].unique())
                pred_labels = sorted(d["predicted_label"].unique())

                matrix = (
                    d.pivot_table(
                        index="gold_label",
                        columns="predicted_label",
                        values="row_pct_within_gold",
                        fill_value=0,
                    )
                    .reindex(index=gold_labels, columns=pred_labels, fill_value=0)
                )

                plt.figure(figsize=(5.5, 4.5))

                image = plt.imshow(matrix.to_numpy())
                plt.colorbar(image, label="% within gold label")

                plt.title(f"Confusion matrix: {task}, {lang}, {quant}")
                plt.xlabel("Predicted label")
                plt.ylabel("Gold label")
                plt.xticks(np.arange(len(pred_labels)), pred_labels)
                plt.yticks(np.arange(len(gold_labels)), gold_labels)

                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        plt.text(
                            j,
                            i,
                            f"{matrix.iloc[i, j]:.0f}",
                            ha="center",
                            va="center",
                            fontsize=9,
                        )

                safe_task = task.replace("/", "_")

                savefig(output_dir, f"confusion_{safe_task}_{lang}_{quant}.png")


def write_index(output_dir: Path) -> None:
    index_path = output_dir / "ualign_figures_index.md"

    index_text = """# UAlign Visual Figures v4

"""

    index_path.write_text(index_text, encoding="utf-8")
    print(f"[SAVED] {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create visual figures for UAlign behavior analysis."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/v3/ualign_behavior_analysis",
        help="Directory with UAlign behavior CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for PNG figures. Default: <input-dir>/figures",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v4",
        help="Version suffix used in CSV file names.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    orientation = pd.read_csv(input_dir / f"ualign_orientation_summary_{args.version}.csv")
    flips = pd.read_csv(input_dir / f"ualign_quantization_flips_summary_{args.version}.csv")
    consistency = pd.read_csv(input_dir / f"ualign_en_uk_consistency_summary_{args.version}.csv")
    confusion = pd.read_csv(input_dir / f"ualign_confusion_matrix_long_{args.version}.csv")

    plot_ethics_unacceptable_rate(orientation, output_dir)
    plot_accuracy_by_task_lang(orientation, output_dir)
    plot_social_label_distribution(orientation, output_dir)
    plot_flip_rates(flips, output_dir)
    plot_en_uk_consistency(consistency, output_dir)
    plot_mean_predicted_label_drift(orientation, output_dir)
    plot_confusion_matrices(confusion, output_dir)
    write_index(output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()