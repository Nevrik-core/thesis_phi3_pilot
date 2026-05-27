# src/analyze_ualign_behavior.py
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


try:
    from config import RESULTS_DIR as CONFIG_RESULTS_DIR
except Exception:
    CONFIG_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


INVALID_LABEL = -1

QUANTIZATION_ORDER = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]

DEFAULT_COMPARISONS = [
    ("BF16", "Q8_0"),
    ("BF16", "Q4_K_M"),
    ("Q8_0", "Q4_K_M"),
]


def label_name(task: str, value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "invalid"

    value = int(value)

    if value == INVALID_LABEL:
        return "invalid"

    if task == "ethics":
        return {
            0: "acceptable",
            1: "unacceptable",
        }.get(value, f"unknown_{value}")

    if task == "social_chemistry":
        return {
            0: "bad",
            1: "expected_or_neutral",
            2: "good",
        }.get(value, f"unknown_{value}")

    return str(value)


def valid_labels_for_task(task: str) -> list[int]:
    if task == "ethics":
        return [0, 1]

    if task == "social_chemistry":
        return [0, 1, 2]

    return []


def quantization_sort_key(value: str) -> int:
    if value in QUANTIZATION_ORDER:
        return QUANTIZATION_ORDER.index(value)
    return 999


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def round_float(value: Any, digits: int = 4) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return round(float(value), digits)


def macro_f1_score(gold: pd.Series, pred: pd.Series, labels: list[int]) -> float:
    f1_values: list[float] = []

    for label in labels:
        tp = int(((gold == label) & (pred == label)).sum())
        fp = int(((gold != label) & (pred == label)).sum())
        fn = int(((gold == label) & (pred != label)).sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_values.append(f1)

    if not f1_values:
        return 0.0

    return round_float(sum(f1_values) / len(f1_values))


def find_latest_input_dir(results_dir: Path) -> Path:
    forced_output_dir = os.getenv("EXPERIMENT_OUTPUT_DIR")
    if forced_output_dir:
        candidate = Path(forced_output_dir)
        if list(candidate.glob("pilot_ualign_details_*_v*.csv")):
            return candidate

    version_dirs = []

    if results_dir.exists():
        for path in results_dir.iterdir():
            if not path.is_dir():
                continue

            if not re.fullmatch(r"v\d+", path.name):
                continue

            if list(path.glob("pilot_ualign_details_*_v*.csv")):
                version_number = int(path.name[1:])
                version_dirs.append((version_number, path))

    if version_dirs:
        version_dirs.sort(key=lambda x: x[0])
        return version_dirs[-1][1]

    if list(results_dir.glob("pilot_ualign_details_*_v*.csv")):
        return results_dir

    raise FileNotFoundError(
        f"No UAlign detail files found in {results_dir} or version subdirectories."
    )


def read_csv_files(paths: list[Path]) -> pd.DataFrame:
    frames = []

    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise RuntimeError("No CSV files were loaded.")

    return pd.concat(frames, ignore_index=True)


def load_ualign_details(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("pilot_ualign_details_*_v*.csv"))

    if not paths:
        paths = sorted(input_dir.rglob("pilot_ualign_details_*_v*.csv"))

    if not paths:
        raise FileNotFoundError(f"No pilot_ualign_details files found in {input_dir}")

    print("\n[LOAD] UAlign detail files:")
    for path in paths:
        print(f"- {path}")

    df = read_csv_files(paths)
    return normalize_details(df)


def normalize_details(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "example_id",
        "task",
        "lang",
        "quantization_name",
        "predicted_label",
        "gold_label",
        "is_valid_answer",
        "is_correct",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in UAlign details: {missing}")

    out = df.copy()

    out["task"] = out["task"].astype(str)
    out["lang"] = out["lang"].astype(str)
    out["example_id"] = out["example_id"].astype(str)
    out["quantization_name"] = out["quantization_name"].astype(str)

    out["gold_label_num"] = pd.to_numeric(out["gold_label"], errors="coerce").astype("Int64")
    out["predicted_label_num"] = pd.to_numeric(
        out["predicted_label"],
        errors="coerce",
    ).astype("Int64")

    out["predicted_label_filled"] = out["predicted_label_num"].fillna(INVALID_LABEL).astype(int)
    out["gold_label_filled"] = out["gold_label_num"].fillna(INVALID_LABEL).astype(int)

    out["is_valid_answer"] = pd.to_numeric(
        out["is_valid_answer"],
        errors="coerce",
    ).fillna(0).astype(int)

    out["is_correct"] = pd.to_numeric(
        out["is_correct"],
        errors="coerce",
    ).fillna(0).astype(int)

    out["predicted_label_name"] = out.apply(
        lambda row: label_name(row["task"], row["predicted_label_filled"]),
        axis=1,
    )
    out["gold_label_name"] = out.apply(
        lambda row: label_name(row["task"], row["gold_label_filled"]),
        axis=1,
    )

    out["quantization_order"] = out["quantization_name"].apply(quantization_sort_key)

    dedupe_cols = [
        "quantization_name",
        "backend_name" if "backend_name" in out.columns else "quantization_name",
        "task",
        "lang",
        "example_id",
    ]

    dedupe_cols = list(dict.fromkeys(dedupe_cols))

    before = len(out)
    out = out.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
    after = len(out)

    if after != before:
        print(f"[WARN] Dropped duplicate rows: {before - after}")

    return out


def build_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "quantization_name",
        "task",
        "lang",
        "predicted_label_filled",
        "predicted_label_name",
    ]

    dist = (
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="count")
    )

    totals = (
        df.groupby(["quantization_name", "task", "lang"], dropna=False)
        .size()
        .reset_index(name="total")
    )

    dist = dist.merge(
        totals,
        on=["quantization_name", "task", "lang"],
        how="left",
    )

    dist["share"] = dist["count"] / dist["total"]
    dist["share_pct"] = (dist["share"] * 100).round(2)

    dist = dist.sort_values(
        ["task", "lang", "quantization_name", "predicted_label_filled"],
        key=lambda s: s.map(quantization_sort_key) if s.name == "quantization_name" else s,
    )

    return dist


def build_gold_distribution(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "task",
        "lang",
        "gold_label_filled",
        "gold_label_name",
    ]

    dist = (
        df.drop_duplicates(subset=["task", "lang", "example_id"])
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="count")
    )

    totals = (
        df.drop_duplicates(subset=["task", "lang", "example_id"])
        .groupby(["task", "lang"], dropna=False)
        .size()
        .reset_index(name="total")
    )

    dist = dist.merge(totals, on=["task", "lang"], how="left")
    dist["share"] = dist["count"] / dist["total"]
    dist["share_pct"] = (dist["share"] * 100).round(2)

    return dist.sort_values(["task", "lang", "gold_label_filled"])


def build_orientation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_cols = ["quantization_name", "task", "lang"]

    for (quantization, task, lang), group in df.groupby(group_cols):
        total = len(group)
        valid = group[group["is_valid_answer"] == 1]
        valid_count = len(valid)

        labels = valid_labels_for_task(task)

        row: dict[str, Any] = {
            "quantization_name": quantization,
            "task": task,
            "lang": lang,
            "n": total,
            "valid_n": valid_count,
            "valid_rate": round_float(valid_count / total),
            "invalid_rate": round_float(1 - valid_count / total),
            "accuracy": round_float(group["is_correct"].mean()),
            "macro_f1_valid_only": macro_f1_score(
                gold=valid["gold_label_filled"],
                pred=valid["predicted_label_filled"],
                labels=labels,
            ),
            "mean_predicted_label_valid": round_float(
                valid["predicted_label_filled"].mean()
            ) if valid_count else 0.0,
            "mean_gold_label": round_float(group["gold_label_filled"].mean()),
        }

        for label in [0, 1, 2]:
            label_count = int((valid["predicted_label_filled"] == label).sum())
            row[f"pred_label_{label}_rate_valid"] = round_float(
                safe_div(label_count, valid_count)
            )

        if task == "ethics":
            row["ethics_unacceptable_rate"] = row["pred_label_1_rate_valid"]
            row["ethics_acceptable_rate"] = row["pred_label_0_rate_valid"]
            row["social_bad_rate"] = None
            row["social_neutral_rate"] = None
            row["social_good_rate"] = None

        elif task == "social_chemistry":
            row["ethics_unacceptable_rate"] = None
            row["ethics_acceptable_rate"] = None
            row["social_bad_rate"] = row["pred_label_0_rate_valid"]
            row["social_neutral_rate"] = row["pred_label_1_rate_valid"]
            row["social_good_rate"] = row["pred_label_2_rate_valid"]

        rows.append(row)

    summary = pd.DataFrame(rows)

    metric_cols = [
        "accuracy",
        "macro_f1_valid_only",
        "valid_rate",
        "invalid_rate",
        "mean_predicted_label_valid",
        "pred_label_0_rate_valid",
        "pred_label_1_rate_valid",
        "pred_label_2_rate_valid",
        "ethics_unacceptable_rate",
        "ethics_acceptable_rate",
        "social_bad_rate",
        "social_neutral_rate",
        "social_good_rate",
    ]

    for reference in ["BF16", "Q8_0"]:
        ref = summary[summary["quantization_name"] == reference][
            ["task", "lang", *metric_cols]
        ].copy()

        if ref.empty:
            continue

        ref = ref.rename(
            columns={col: f"{col}_{reference}" for col in metric_cols}
        )

        summary = summary.merge(ref, on=["task", "lang"], how="left")

        for col in metric_cols:
            ref_col = f"{col}_{reference}"
            if ref_col in summary.columns:
                summary[f"{col}_delta_vs_{reference}"] = (
                    summary[col] - summary[ref_col]
                ).round(4)

    summary["quantization_order"] = summary["quantization_name"].apply(quantization_sort_key)

    return summary.sort_values(["task", "lang", "quantization_order"]).drop(
        columns=["quantization_order"]
    )


def build_confusion_matrices(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (quantization, task, lang), group in df.groupby(
        ["quantization_name", "task", "lang"]
    ):
        labels = valid_labels_for_task(task)
        pred_labels = [INVALID_LABEL, *labels]

        for gold_label in labels:
            gold_group = group[group["gold_label_filled"] == gold_label]
            gold_total = len(gold_group)

            for pred_label in pred_labels:
                count = int((gold_group["predicted_label_filled"] == pred_label).sum())

                rows.append(
                    {
                        "quantization_name": quantization,
                        "task": task,
                        "lang": lang,
                        "gold_label": gold_label,
                        "gold_label_name": label_name(task, gold_label),
                        "predicted_label": pred_label,
                        "predicted_label_name": label_name(task, pred_label),
                        "count": count,
                        "row_total_for_gold": gold_total,
                        "row_pct_within_gold": round_float(
                            safe_div(count, gold_total) * 100,
                            digits=2,
                        ),
                        "total_pct": round_float(
                            safe_div(count, len(group)) * 100,
                            digits=2,
                        ),
                    }
                )

    out = pd.DataFrame(rows)
    out["quantization_order"] = out["quantization_name"].apply(quantization_sort_key)

    return out.sort_values(
        [
            "task",
            "lang",
            "quantization_order",
            "gold_label",
            "predicted_label",
        ]
    ).drop(columns=["quantization_order"])


def correctness_change(ref_correct: int, target_correct: int) -> str:
    if ref_correct == 1 and target_correct == 1:
        return "correct_to_correct"
    if ref_correct == 1 and target_correct == 0:
        return "correct_to_wrong"
    if ref_correct == 0 and target_correct == 1:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def orientation_change(task: str, ref_pred: int, target_pred: int) -> str:
    if ref_pred == INVALID_LABEL and target_pred == INVALID_LABEL:
        return "invalid_to_invalid"
    if ref_pred == INVALID_LABEL:
        return "invalid_to_valid"
    if target_pred == INVALID_LABEL:
        return "valid_to_invalid"

    delta = target_pred - ref_pred

    if delta == 0:
        return "same_label"

    if task == "ethics":
        if delta > 0:
            return "became_more_strict_or_unacceptable"
        return "became_more_permissive_or_acceptable"

    if task == "social_chemistry":
        if delta > 0:
            return "became_more_positive"
        return "became_more_negative"

    return "changed_label"


def build_quantization_flips(
    df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_cols = ["task", "lang", "example_id"]

    metadata_cols = [
        col for col in [
            "gold_label_filled",
            "gold_label_name",
            "label_space",
            "text",
        ]
        if col in df.columns
    ]

    base_meta = (
        df.sort_values(["quantization_order"])
        .groupby(index_cols, as_index=False)
        .first()[index_cols + metadata_cols]
    )

    pred_wide = df.pivot_table(
        index=index_cols,
        columns="quantization_name",
        values="predicted_label_filled",
        aggfunc="first",
    ).reset_index()

    correct_wide = df.pivot_table(
        index=index_cols,
        columns="quantization_name",
        values="is_correct",
        aggfunc="first",
    ).reset_index()

    valid_wide = df.pivot_table(
        index=index_cols,
        columns="quantization_name",
        values="is_valid_answer",
        aggfunc="first",
    ).reset_index()

    pred_wide = base_meta.merge(pred_wide, on=index_cols, how="left")

    rows: list[dict[str, Any]] = []

    available_quants = set(df["quantization_name"].unique())

    for ref_quant, target_quant in comparisons:
        if ref_quant not in available_quants or target_quant not in available_quants:
            print(
                f"[WARN] Skipping comparison {ref_quant} -> {target_quant}: "
                f"missing one of the quantizations."
            )
            continue

        merged = pred_wide.copy()

        merged = merged.merge(
            correct_wide[index_cols + [ref_quant, target_quant]].rename(
                columns={
                    ref_quant: f"{ref_quant}_correct",
                    target_quant: f"{target_quant}_correct",
                }
            ),
            on=index_cols,
            how="left",
        )

        merged = merged.merge(
            valid_wide[index_cols + [ref_quant, target_quant]].rename(
                columns={
                    ref_quant: f"{ref_quant}_valid",
                    target_quant: f"{target_quant}_valid",
                }
            ),
            on=index_cols,
            how="left",
        )

        for _, row in merged.iterrows():
            ref_pred = row.get(ref_quant)
            target_pred = row.get(target_quant)

            if pd.isna(ref_pred) or pd.isna(target_pred):
                continue

            ref_pred = int(ref_pred)
            target_pred = int(target_pred)

            ref_correct = int(row.get(f"{ref_quant}_correct", 0))
            target_correct = int(row.get(f"{target_quant}_correct", 0))

            task = str(row["task"])

            rows.append(
                {
                    "reference_quantization": ref_quant,
                    "target_quantization": target_quant,
                    "task": task,
                    "lang": row["lang"],
                    "example_id": row["example_id"],
                    "gold_label": int(row["gold_label_filled"]),
                    "gold_label_name": row["gold_label_name"],
                    "reference_predicted_label": ref_pred,
                    "reference_predicted_label_name": label_name(task, ref_pred),
                    "target_predicted_label": target_pred,
                    "target_predicted_label_name": label_name(task, target_pred),
                    "label_changed": int(ref_pred != target_pred),
                    "label_delta": (
                        target_pred - ref_pred
                        if ref_pred != INVALID_LABEL and target_pred != INVALID_LABEL
                        else None
                    ),
                    "reference_is_correct": ref_correct,
                    "target_is_correct": target_correct,
                    "correctness_change": correctness_change(
                        ref_correct,
                        target_correct,
                    ),
                    "orientation_change": orientation_change(
                        task,
                        ref_pred,
                        target_pred,
                    ),
                    "label_space": row.get("label_space", ""),
                    "text": row.get("text", ""),
                }
            )

    flips = pd.DataFrame(rows)

    if flips.empty:
        return flips, pd.DataFrame()

    summary = (
        flips.groupby(
            [
                "reference_quantization",
                "target_quantization",
                "task",
                "lang",
            ],
            dropna=False,
        )
        .agg(
            n=("example_id", "count"),
            label_changed_count=("label_changed", "sum"),
            reference_accuracy=("reference_is_correct", "mean"),
            target_accuracy=("target_is_correct", "mean"),
            correct_to_wrong_count=(
                "correctness_change",
                lambda s: int((s == "correct_to_wrong").sum()),
            ),
            wrong_to_correct_count=(
                "correctness_change",
                lambda s: int((s == "wrong_to_correct").sum()),
            ),
            same_label_count=(
                "orientation_change",
                lambda s: int((s == "same_label").sum()),
            ),
            valid_to_invalid_count=(
                "orientation_change",
                lambda s: int((s == "valid_to_invalid").sum()),
            ),
            invalid_to_valid_count=(
                "orientation_change",
                lambda s: int((s == "invalid_to_valid").sum()),
            ),
        )
        .reset_index()
    )

    summary["label_changed_rate"] = (
        summary["label_changed_count"] / summary["n"]
    ).round(4)

    summary["accuracy_delta"] = (
        summary["target_accuracy"] - summary["reference_accuracy"]
    ).round(4)

    summary["correct_to_wrong_rate"] = (
        summary["correct_to_wrong_count"] / summary["n"]
    ).round(4)

    summary["wrong_to_correct_rate"] = (
        summary["wrong_to_correct_count"] / summary["n"]
    ).round(4)

    orientation_counts = (
        flips.groupby(
            [
                "reference_quantization",
                "target_quantization",
                "task",
                "lang",
                "orientation_change",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    orientation_pivot = orientation_counts.pivot_table(
        index=[
            "reference_quantization",
            "target_quantization",
            "task",
            "lang",
        ],
        columns="orientation_change",
        values="count",
        fill_value=0,
    ).reset_index()

    summary = summary.merge(
        orientation_pivot,
        on=[
            "reference_quantization",
            "target_quantization",
            "task",
            "lang",
        ],
        how="left",
    )

    return flips, summary


def language_stability_category(row: pd.Series) -> str:
    uk_valid = row["uk_predicted_label"] != INVALID_LABEL
    en_valid = row["en_predicted_label"] != INVALID_LABEL

    if not uk_valid or not en_valid:
        return "invalid_in_one_or_both"

    labels_match = row["uk_predicted_label"] == row["en_predicted_label"]
    uk_correct = row["uk_is_correct"] == 1
    en_correct = row["en_is_correct"] == 1

    if labels_match and uk_correct and en_correct:
        return "same_label_both_correct"

    if labels_match and not uk_correct and not en_correct:
        return "same_label_both_wrong"

    if labels_match:
        return "same_label_mixed_correctness"

    if uk_correct and not en_correct:
        return "different_label_uk_correct_en_wrong"

    if en_correct and not uk_correct:
        return "different_label_en_correct_uk_wrong"

    return "different_label_both_wrong"


def build_cross_language_consistency(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_cols = ["quantization_name", "task", "example_id"]

    pred = df.pivot_table(
        index=index_cols,
        columns="lang",
        values="predicted_label_filled",
        aggfunc="first",
    ).reset_index()

    correct = df.pivot_table(
        index=index_cols,
        columns="lang",
        values="is_correct",
        aggfunc="first",
    ).reset_index()

    valid = df.pivot_table(
        index=index_cols,
        columns="lang",
        values="is_valid_answer",
        aggfunc="first",
    ).reset_index()

    gold = (
        df.groupby(index_cols, as_index=False)
        .agg(
            gold_label=("gold_label_filled", "first"),
            gold_label_name=("gold_label_name", "first"),
            label_space=("label_space", "first") if "label_space" in df.columns else ("task", "first"),
        )
    )

    text_uk = (
        df[df["lang"] == "uk"]
        .groupby(index_cols, as_index=False)
        .agg(uk_text=("text", "first"))
        if "text" in df.columns
        else pd.DataFrame()
    )

    text_en = (
        df[df["lang"] == "en"]
        .groupby(index_cols, as_index=False)
        .agg(en_text=("text", "first"))
        if "text" in df.columns
        else pd.DataFrame()
    )

    required_langs = {"uk", "en"}
    if not required_langs.issubset(set(pred.columns)):
        raise ValueError(
            "Cross-language consistency requires both 'uk' and 'en' rows."
        )

    out = gold.merge(pred, on=index_cols, how="left")
    out = out.merge(
        correct.rename(columns={"uk": "uk_is_correct", "en": "en_is_correct"}),
        on=index_cols,
        how="left",
    )
    out = out.merge(
        valid.rename(columns={"uk": "uk_is_valid", "en": "en_is_valid"}),
        on=index_cols,
        how="left",
    )

    out = out.rename(
        columns={
            "uk": "uk_predicted_label",
            "en": "en_predicted_label",
        }
    )

    if not text_uk.empty:
        out = out.merge(text_uk, on=index_cols, how="left")

    if not text_en.empty:
        out = out.merge(text_en, on=index_cols, how="left")

    out["uk_predicted_label"] = out["uk_predicted_label"].fillna(INVALID_LABEL).astype(int)
    out["en_predicted_label"] = out["en_predicted_label"].fillna(INVALID_LABEL).astype(int)

    out["uk_predicted_label_name"] = out.apply(
        lambda row: label_name(row["task"], row["uk_predicted_label"]),
        axis=1,
    )
    out["en_predicted_label_name"] = out.apply(
        lambda row: label_name(row["task"], row["en_predicted_label"]),
        axis=1,
    )

    out["both_valid"] = (
        (out["uk_predicted_label"] != INVALID_LABEL)
        & (out["en_predicted_label"] != INVALID_LABEL)
    ).astype(int)

    out["labels_match_on_valid_pair"] = (
        (out["both_valid"] == 1)
        & (out["uk_predicted_label"] == out["en_predicted_label"])
    ).astype(int)

    out["raw_labels_match_including_invalid"] = (
        out["uk_predicted_label"] == out["en_predicted_label"]
    ).astype(int)

    out["en_minus_uk_label"] = out.apply(
        lambda row: (
            row["en_predicted_label"] - row["uk_predicted_label"]
            if row["both_valid"] == 1
            else None
        ),
        axis=1,
    )

    out["language_stability_category"] = out.apply(
        language_stability_category,
        axis=1,
    )

    summary = (
        out.groupby(["quantization_name", "task"], dropna=False)
        .agg(
            n=("example_id", "count"),
            both_valid_rate=("both_valid", "mean"),
            label_match_rate_on_valid_pairs=(
                "labels_match_on_valid_pair",
                lambda s: safe_div(s.sum(), out.loc[s.index, "both_valid"].sum()),
            ),
            raw_label_match_rate=("raw_labels_match_including_invalid", "mean"),
            uk_accuracy=("uk_is_correct", "mean"),
            en_accuracy=("en_is_correct", "mean"),
            mean_en_minus_uk_label=("en_minus_uk_label", "mean"),
        )
        .reset_index()
    )

    summary["en_minus_uk_accuracy"] = (
        summary["en_accuracy"] - summary["uk_accuracy"]
    ).round(4)

    for col in [
        "both_valid_rate",
        "label_match_rate_on_valid_pairs",
        "raw_label_match_rate",
        "uk_accuracy",
        "en_accuracy",
        "mean_en_minus_uk_label",
    ]:
        summary[col] = summary[col].round(4)

    category_counts = (
        out.groupby(
            [
                "quantization_name",
                "task",
                "language_stability_category",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    category_pivot = category_counts.pivot_table(
        index=["quantization_name", "task"],
        columns="language_stability_category",
        values="count",
        fill_value=0,
    ).reset_index()

    summary = summary.merge(
        category_pivot,
        on=["quantization_name", "task"],
        how="left",
    )

    for reference in ["BF16", "Q8_0"]:
        ref = summary[summary["quantization_name"] == reference][
            [
                "task",
                "label_match_rate_on_valid_pairs",
                "raw_label_match_rate",
                "en_minus_uk_accuracy",
                "mean_en_minus_uk_label",
            ]
        ].copy()

        if ref.empty:
            continue

        ref = ref.rename(
            columns={
                "label_match_rate_on_valid_pairs": f"label_match_rate_on_valid_pairs_{reference}",
                "raw_label_match_rate": f"raw_label_match_rate_{reference}",
                "en_minus_uk_accuracy": f"en_minus_uk_accuracy_{reference}",
                "mean_en_minus_uk_label": f"mean_en_minus_uk_label_{reference}",
            }
        )

        summary = summary.merge(ref, on="task", how="left")

        for metric in [
            "label_match_rate_on_valid_pairs",
            "raw_label_match_rate",
            "en_minus_uk_accuracy",
            "mean_en_minus_uk_label",
        ]:
            ref_col = f"{metric}_{reference}"
            summary[f"{metric}_delta_vs_{reference}"] = (
                summary[metric] - summary[ref_col]
            ).round(4)

    summary["quantization_order"] = summary["quantization_name"].apply(quantization_sort_key)

    summary = summary.sort_values(["task", "quantization_order"]).drop(
        columns=["quantization_order"]
    )

    return out, summary


def build_inspection_cases(
    flips: pd.DataFrame,
    consistency: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    if not flips.empty:
        important_flips = flips[
            (flips["label_changed"] == 1)
            | (flips["correctness_change"].isin(["correct_to_wrong", "wrong_to_correct"]))
        ].copy()

        important_flips["case_type"] = "quantization_flip"
        frames.append(important_flips)

    if not consistency.empty:
        unstable = consistency[
            (consistency["both_valid"] == 1)
            & (consistency["labels_match_on_valid_pair"] == 0)
        ].copy()

        unstable["case_type"] = "en_uk_inconsistent"
        frames.append(unstable)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data._"

    small = df.head(max_rows).copy()

    try:
        return small.to_markdown(index=False)
    except Exception:
        return small.to_string(index=False)


def write_markdown_report(
    output_dir: Path,
    version_label: str,
    df: pd.DataFrame,
    label_distribution: pd.DataFrame,
    orientation_summary: pd.DataFrame,
    flip_summary: pd.DataFrame,
    consistency_summary: pd.DataFrame,
) -> Path:
    report_path = output_dir / f"ualign_behavior_report_{version_label}.md"

    selected_orientation_cols = [
        col for col in [
            "quantization_name",
            "task",
            "lang",
            "n",
            "accuracy",
            "macro_f1_valid_only",
            "invalid_rate",
            "mean_predicted_label_valid",
            "ethics_unacceptable_rate",
            "social_bad_rate",
            "social_neutral_rate",
            "social_good_rate",
            "accuracy_delta_vs_BF16",
            "mean_predicted_label_valid_delta_vs_BF16",
        ]
        if col in orientation_summary.columns
    ]

    selected_flip_cols = [
        col for col in [
            "reference_quantization",
            "target_quantization",
            "task",
            "lang",
            "n",
            "label_changed_rate",
            "accuracy_delta",
            "correct_to_wrong_rate",
            "wrong_to_correct_rate",
            "became_more_strict_or_unacceptable",
            "became_more_permissive_or_acceptable",
            "became_more_positive",
            "became_more_negative",
        ]
        if col in flip_summary.columns
    ]

    selected_consistency_cols = [
        col for col in [
            "quantization_name",
            "task",
            "n",
            "both_valid_rate",
            "label_match_rate_on_valid_pairs",
            "raw_label_match_rate",
            "uk_accuracy",
            "en_accuracy",
            "en_minus_uk_accuracy",
            "mean_en_minus_uk_label",
            "label_match_rate_on_valid_pairs_delta_vs_BF16",
        ]
        if col in consistency_summary.columns
    ]

    text = f"""# UAlign Behavior Analysis Report

## 1. Scope

This report analyzes UAlign behavior beyond accuracy.

The goal is to check whether quantization changes the model's label-selection behavior on ethics/social-norm tasks:

- predicted label distribution;
- behavior/orientation drift between quantization modes;
- BF16 -> Q8_0 -> Q4_K_M answer changes;
- cross-lingual EN-UA stability on parallel examples.

Important: this does not measure the model's "true morality". It measures operational behavior under fixed prompts, fixed labels, and the UAlign label taxonomy.

## 2. Input data

Rows loaded: {len(df)}

Quantization modes:
{", ".join(sorted(df["quantization_name"].unique(), key=quantization_sort_key))}

Tasks:
{", ".join(sorted(df["task"].unique()))}

Languages:
{", ".join(sorted(df["lang"].unique()))}

## 3. Orientation summary

Interpretation:

- ETHICS: higher `ethics_unacceptable_rate` means the model more often classifies situations as unacceptable.
- Social Chemistry: higher `social_bad_rate` means more negative judgment; higher `social_good_rate` means more positive judgment.
- `mean_predicted_label_valid_delta_vs_BF16` shows the direction of drift compared with BF16.

{df_to_markdown(orientation_summary[selected_orientation_cols], max_rows=40)}

## 4. Quantization flips summary

Interpretation:

- `label_changed_rate` shows how often the predicted label changes between two quantization modes.
- `correct_to_wrong_rate` is potential degradation hidden by aggregate metrics.
- `wrong_to_correct_rate` is potential improvement or noise.
- Directional columns show whether labels became stricter/more permissive or more positive/negative.

{df_to_markdown(flip_summary[selected_flip_cols], max_rows=40)}

## 5. EN-UA consistency summary

Interpretation:

- `label_match_rate_on_valid_pairs` shows how often Ukrainian and English versions of the same example receive the same valid label.
- A drop after quantization suggests lower cross-lingual behavioral stability.
- `en_minus_uk_accuracy` shows whether English remains easier than Ukrainian for this task.

{df_to_markdown(consistency_summary[selected_consistency_cols], max_rows=40)}

## 6. Main output files

- `ualign_label_distribution_{version_label}.csv`
- `ualign_gold_distribution_{version_label}.csv`
- `ualign_orientation_summary_{version_label}.csv`
- `ualign_confusion_matrix_long_{version_label}.csv`
- `ualign_quantization_flips_{version_label}.csv`
- `ualign_quantization_flips_summary_{version_label}.csv`
- `ualign_en_uk_consistency_{version_label}.csv`
- `ualign_en_uk_consistency_summary_{version_label}.csv`
- `ualign_behavior_inspection_cases_{version_label}.csv`

## 7. Suggested thesis wording

UAlign results were analyzed not only by aggregate accuracy, but also by the distribution of predicted labels, label flips across quantization modes, and cross-lingual consistency between English and Ukrainian parallel examples. This analysis allows us to detect behavioral shifts that can remain hidden when only accuracy is considered.
"""

    report_path.write_text(text, encoding="utf-8")
    return report_path


def parse_comparisons(value: str | None) -> list[tuple[str, str]]:
    if not value:
        return DEFAULT_COMPARISONS

    comparisons = []

    for part in value.split(","):
        part = part.strip()

        if not part:
            continue

        if "->" not in part:
            raise ValueError(
                "Comparison format must be like 'BF16->Q8_0,BF16->Q4_K_M'"
            )

        left, right = part.split("->", 1)
        comparisons.append((left.strip(), right.strip()))

    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep UAlign behavior analysis for quantized Phi-4-mini runs."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory with pilot_ualign_details_* CSV files. Default: latest results/v* directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for analysis outputs. Default: <input-dir>/ualign_behavior_analysis.",
    )

    parser.add_argument(
        "--version",
        type=str,
        default=os.getenv("BEHAVIOR_ANALYSIS_VERSION", "v4"),
        help="Output version label. Default: v4.",
    )

    parser.add_argument(
        "--comparisons",
        type=str,
        default=None,
        help="Comma-separated comparisons, e.g. 'BF16->Q8_0,BF16->Q4_K_M,Q8_0->Q4_K_M'.",
    )

    args = parser.parse_args()

    results_dir = Path(CONFIG_RESULTS_DIR)

    input_dir = Path(args.input_dir) if args.input_dir else find_latest_input_dir(results_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_dir / "ualign_behavior_analysis"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = parse_comparisons(args.comparisons)

    print("\n=== UALIGN BEHAVIOR ANALYSIS ===")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Output version: {args.version}")
    print(f"Comparisons: {comparisons}")

    df = load_ualign_details(input_dir)

    label_distribution = build_label_distribution(df)
    gold_distribution = build_gold_distribution(df)
    orientation_summary = build_orientation_summary(df)
    confusion = build_confusion_matrices(df)

    flips, flip_summary = build_quantization_flips(
        df=df,
        comparisons=comparisons,
    )

    consistency, consistency_summary = build_cross_language_consistency(df)

    inspection_cases = build_inspection_cases(
        flips=flips,
        consistency=consistency,
    )

    version = args.version

    outputs = {
        f"ualign_label_distribution_{version}.csv": label_distribution,
        f"ualign_gold_distribution_{version}.csv": gold_distribution,
        f"ualign_orientation_summary_{version}.csv": orientation_summary,
        f"ualign_confusion_matrix_long_{version}.csv": confusion,
        f"ualign_quantization_flips_{version}.csv": flips,
        f"ualign_quantization_flips_summary_{version}.csv": flip_summary,
        f"ualign_en_uk_consistency_{version}.csv": consistency,
        f"ualign_en_uk_consistency_summary_{version}.csv": consistency_summary,
        f"ualign_behavior_inspection_cases_{version}.csv": inspection_cases,
    }

    for filename, table in outputs.items():
        path = output_dir / filename
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {path}")

    report_path = write_markdown_report(
        output_dir=output_dir,
        version_label=version,
        df=df,
        label_distribution=label_distribution,
        orientation_summary=orientation_summary,
        flip_summary=flip_summary,
        consistency_summary=consistency_summary,
    )

    print(f"[SAVED] {report_path}")

    print("\n=== QUICK ORIENTATION SUMMARY ===")
    print(
        orientation_summary[
            [
                col for col in [
                    "quantization_name",
                    "task",
                    "lang",
                    "accuracy",
                    "macro_f1_valid_only",
                    "invalid_rate",
                    "mean_predicted_label_valid",
                    "ethics_unacceptable_rate",
                    "social_bad_rate",
                    "social_neutral_rate",
                    "social_good_rate",
                    "mean_predicted_label_valid_delta_vs_BF16",
                ]
                if col in orientation_summary.columns
            ]
        ].to_string(index=False)
    )

    print("\n=== QUICK FLIP SUMMARY ===")
    if flip_summary.empty:
        print("No flip summary.")
    else:
        print(flip_summary.to_string(index=False))

    print("\n=== QUICK EN-UA CONSISTENCY SUMMARY ===")
    print(consistency_summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()