import os
from pathlib import Path

import pandas as pd


def next_experiment_version(results_dir: Path, prefix: str) -> int:
    existing_versions = []

    for path in results_dir.rglob(f"{prefix}_v*.csv"):
        stem = path.stem

        try:
            version_part = stem.rsplit("_v", 1)[1]
            existing_versions.append(int(version_part))
        except (IndexError, ValueError):
            continue

    if not existing_versions:
        return 1

    return max(existing_versions) + 1


def resolve_experiment_version(results_dir: Path, prefix: str) -> str:
    forced_version = os.getenv("EXPERIMENT_VERSION")

    if forced_version:
        return forced_version

    version_num = next_experiment_version(results_dir, prefix)
    return f"v{version_num}"


def make_experiment_output_dir(results_dir: Path, experiment_version: str) -> Path:
    forced_output_dir = os.getenv("EXPERIMENT_OUTPUT_DIR")

    if forced_output_dir:
        output_dir = Path(forced_output_dir)
    else:
        output_dir = results_dir / experiment_version

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_mean(df: pd.DataFrame, column: str, digits: int = 4):
    if column not in df.columns or len(df) == 0:
        return 0.0

    values = df[column].dropna()

    if values.empty:
        return 0.0

    return round(float(values.mean()), digits)


def safe_max(df: pd.DataFrame, column: str, digits: int = 4):
    if column not in df.columns or len(df) == 0:
        return 0.0

    values = df[column].dropna()

    if values.empty:
        return 0.0

    return round(float(values.max()), digits)


def first_non_null(df: pd.DataFrame, column: str):
    if column not in df.columns or len(df) == 0:
        return None

    values = df[column].dropna()

    if values.empty:
        return None

    return values.iloc[0]