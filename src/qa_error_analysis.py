# src/qa_error_analysis.py
import os
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR
from eval_utils import next_experiment_version


OUTPUT_ROOT = RESULTS_DIR / "qa_error_analysis"

QA_DETAILS_PATTERN = "pilot_qa_details_*_v*.csv"


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        return ""

    return " ".join(value.lower().strip().split())


def split_gold_answers(value: str) -> list[str]:
    if not isinstance(value, str):
        return []

    return [part.strip() for part in value.split("|") if part.strip()]


def has_gold_inside_prediction(prediction: str, gold_answers: list[str]) -> bool:
    pred_norm = normalize_text(prediction)

    if not pred_norm:
        return False

    for gold in gold_answers:
        gold_norm = normalize_text(gold)

        if gold_norm and gold_norm in pred_norm and pred_norm != gold_norm:
            return True

    return False


def suggest_error_category(row: pd.Series) -> str:
    prediction = row.get("prediction", "")
    gold_answers_raw = row.get("gold_answers", "")
    exact_match = float(row.get("exact_match", 0) or 0)
    f1 = float(row.get("f1", 0) or 0)

    gold_answers = split_gold_answers(gold_answers_raw)

    if not gold_answers:
        return "unanswerable_or_dataset_issue"

    if not isinstance(prediction, str) or not prediction.strip():
        return "wrong_answer"

    if exact_match == 1:
        return "correct_meaning"

    if has_gold_inside_prediction(prediction, gold_answers):
        return "extra_text"

    if f1 >= 0.8:
        return "correct_meaning"

    if 0.0 < f1 < 0.8:
        return "format_mismatch"

    return "wrong_answer"


def resolve_output_context() -> tuple[Path, Path, str]:
    forced_version = os.getenv("EXPERIMENT_VERSION")
    forced_output_dir = os.getenv("EXPERIMENT_OUTPUT_DIR")

    if forced_output_dir:
        search_root = Path(forced_output_dir)
        experiment_version = forced_version or search_root.name
        output_dir = search_root / "qa_error_analysis"
    else:
        search_root = RESULTS_DIR
        output_prefix = "qa_error_analysis_uk"
        version_num = next_experiment_version(OUTPUT_ROOT, output_prefix)
        experiment_version = f"v{version_num}"
        output_dir = OUTPUT_ROOT / experiment_version

    output_dir.mkdir(parents=True, exist_ok=True)

    return search_root, output_dir, experiment_version


def find_qa_details(search_root: Path) -> list[Path]:
    paths = [
        path
        for path in search_root.rglob(QA_DETAILS_PATTERN)
        if "qa_error_analysis" not in str(path)
        and "week3_figures" not in str(path)
    ]

    return sorted(paths)


def read_details(paths: list[Path]) -> pd.DataFrame:
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


def keep_latest_per_backend_lang_example(df: pd.DataFrame) -> pd.DataFrame:
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
            "backend_name",
            "quantization_name",
            "model_role",
            "lang",
            "example_id",
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


def ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    optional_defaults = {
        "model_name": "unknown",
        "backend_name": "unknown",
        "model_role": "unknown",
    }

    for column, default_value in optional_defaults.items():
        if column not in out.columns:
            out[column] = default_value
        else:
            out[column] = out[column].fillna(default_value)

    return out


def build_error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = ensure_optional_columns(df)

    uk_df = df[df["lang"] == "uk"].copy()

    if uk_df.empty:
        return uk_df

    required_cols = [
        "example_id",
        "lang",
        "model_name",
        "backend_name",
        "quantization_name",
        "model_role",
        "question",
        "prediction",
        "gold_answers",
        "exact_match",
        "f1",
    ]

    missing = [col for col in required_cols if col not in uk_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in QA details: {missing}")

    out = uk_df[required_cols].copy()

    out["error_category"] = out.apply(suggest_error_category, axis=1)
    out["error_notes"] = ""

    out["needs_manual_review"] = out["exact_match"].astype(float) != 1.0
    out["is_low_f1"] = out["f1"].astype(float) < 0.5

    out = out.sort_values(
        by=[
            "backend_name",
            "quantization_name",
            "needs_manual_review",
            "f1",
            "example_id",
        ],
        ascending=[True, True, False, True, True],
    )

    return out


def build_summary(error_df: pd.DataFrame) -> pd.DataFrame:
    if error_df.empty:
        return pd.DataFrame()

    group_cols = [
        "backend_name",
        "quantization_name",
        "model_role",
        "error_category",
    ]

    summary_df = (
        error_df.groupby(group_cols, as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "count"})
    )

    return summary_df.sort_values(
        by=[
            "backend_name",
            "quantization_name",
            "model_role",
            "error_category",
        ]
    )


def main():
    search_root, output_dir, experiment_version = resolve_output_context()

    print("\n=== QA ERROR ANALYSIS ===")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Search root: {search_root}")
    print(f"Output dir: {output_dir}")
    print(f"Version: {experiment_version}")

    paths = find_qa_details(search_root)

    print(f"\nFound QA details files: {len(paths)}")
    for path in paths:
        print(f"  - {path}")

    details_df = read_details(paths)

    if details_df.empty:
        raise FileNotFoundError("No QA details files found.")

    details_df = ensure_optional_columns(details_df)
    details_df = keep_latest_per_backend_lang_example(details_df)

    error_df = build_error_analysis(details_df)

    output_prefix = "qa_error_analysis_uk"

    output_path = output_dir / f"{output_prefix}_{experiment_version}.csv"
    error_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary_df = build_summary(error_df)

    summary_path = output_dir / f"{output_prefix}_summary_{experiment_version}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== ERROR CATEGORY SUMMARY ===")
    print(summary_df.to_string(index=False))

    print(f"\nSaved error analysis to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()