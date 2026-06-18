from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


QUANTIZATION_ORDER = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
QUANTIZATION_INDEX = {name: i for i, name in enumerate(QUANTIZATION_ORDER)}

# Used only for Ukrainian content-adjusted QA evaluation.
UK_STOPWORDS = {
    "у", "в", "на", "з", "із", "зі", "до", "від", "за", "про", "через",
    "під", "при", "над", "без", "по", "для", "та", "і", "й", "а", "але", "або",
}


def quantization_sort_key(value: str) -> int:
    return QUANTIZATION_INDEX.get(str(value), 999)


def round_or_none(value: Any, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def find_qa_detail_files(input_dir: Path, version: str) -> list[Path]:
    paths = sorted(
        path
        for path in input_dir.glob(f"pilot_qa_details_*_{version}.csv")
        if "qa_error_analysis" not in str(path)
        and "week3_figures" not in str(path)
        and "speed_token_analysis" not in str(path)
    )

    if not paths:
        raise FileNotFoundError(
            f"No QA detail files found in {input_dir} for version {version}"
        )

    return paths


def read_qa_details(input_dir: Path, version: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    print("\n[LOAD] QA detail files:")

    for path in find_qa_detail_files(input_dir, version):
        print(f"- {path.name}")
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    required = {
        "lang",
        "quantization_name",
        "prediction",
        "gold_answers",
        "exact_match",
        "f1",
    }

    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(out.columns)}"
        )

    out["lang"] = out["lang"].astype(str).str.lower().str.strip()
    out["quantization_name"] = out["quantization_name"].astype(str).str.strip()
    out["exact_match"] = pd.to_numeric(out["exact_match"], errors="coerce").fillna(0)
    out["f1"] = pd.to_numeric(out["f1"], errors="coerce").fillna(0)

    dedupe_cols = [
        col
        for col in ["backend_name", "quantization_name", "lang", "example_id"]
        if col in out.columns
    ]

    if dedupe_cols:
        before = len(out)
        out = out.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
        after = len(out)

        if after != before:
            print(f"[WARN] Dropped duplicate QA rows: {before - after}")

    return out


# -----------------------------------------------------------------------------
# Ukrainian content-adjusted matching
# -----------------------------------------------------------------------------
def normalize_basic(value: str) -> str:
    if not isinstance(value, str):
        return ""

    text = value.lower().strip()
    text = text.replace("’", "'").replace("ʼ", "'").replace("`", "'")
    text = text.replace("年", "")

    text = re.sub(r"[«»„“”’…—–−]", " ", text)
    text = re.sub(r"[-]", " ", text)
    text = re.sub(r"[^\w\s\.'’]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(value: str) -> list[str]:
    text = normalize_basic(value)

    return re.findall(
        r"[a-zа-яіїєґ0-9]+(?:[.'’][a-zа-яіїєґ0-9]+)?",
        text,
        flags=re.IGNORECASE,
    )


def remove_uk_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in UK_STOPWORDS]


def canonical_uk(value: str) -> str:
    return " ".join(remove_uk_stopwords(tokenize(value)))


def split_gold_answers(value: str) -> list[str]:
    if not isinstance(value, str):
        return []

    return [part.strip() for part in value.split("|") if part.strip()]


def extract_numbers(value: str) -> set[str]:
    if not isinstance(value, str):
        return set()

    normalized = value.replace(",", ".")
    return set(re.findall(r"\d+(?:\.\d+)?", normalized))


def token_coverage_uk(prediction: str, gold: str, threshold: float = 0.78) -> float:
    pred_tokens = remove_uk_stopwords(tokenize(prediction))
    gold_tokens = remove_uk_stopwords(tokenize(gold))

    if not pred_tokens or not gold_tokens:
        return 0.0

    matched = 0

    for gold_token in gold_tokens:
        best = max(
            SequenceMatcher(None, gold_token, pred_token).ratio()
            for pred_token in pred_tokens
        )

        if best >= threshold:
            matched += 1

    return matched / len(gold_tokens)


def fuzzy_uk_content_match(prediction: str, gold: str) -> bool:
    pred_canonical = canonical_uk(prediction)
    gold_canonical = canonical_uk(gold)

    if not pred_canonical or not gold_canonical:
        return False

    pred_numbers = extract_numbers(prediction)
    gold_numbers = extract_numbers(gold)

    # Conservative numeric rule:
    # all numbers from the reference answer must appear in the prediction.
    if gold_numbers and gold_numbers.issubset(pred_numbers):
        return True

    if pred_canonical == gold_canonical:
        return True

    if gold_canonical in pred_canonical:
        return True

    if pred_canonical in gold_canonical:
        return True

    ratio = SequenceMatcher(None, pred_canonical, gold_canonical).ratio()


    if ratio >= 0.82:
        return True

    coverage = token_coverage_uk(prediction, gold, threshold=0.78)
    gold_token_count = len(remove_uk_stopwords(tokenize(gold)))

    if gold_token_count >= 2 and coverage >= 0.75:
        return True

    if gold_token_count == 1 and coverage == 1.0 and ratio >= 0.65:
        return True

    return False


def has_any_uk_soft_match(row: pd.Series) -> int:
    prediction = row.get("prediction", "")
    gold_answers = split_gold_answers(row.get("gold_answers", ""))

    for gold in gold_answers:
        if fuzzy_uk_content_match(prediction, gold):
            return 1

    return 0


def add_uk_content_adjusted_column(
    details: pd.DataFrame,
    f1_threshold: float,
) -> pd.DataFrame:
    out = details.copy()

    out["uk_soft_match"] = 0
    out["uk_content_adjusted_correct"] = 0

    uk_mask = out["lang"] == "uk"

    out.loc[uk_mask, "uk_soft_match"] = out.loc[uk_mask].apply(
        has_any_uk_soft_match,
        axis=1,
    )

    out.loc[uk_mask, "uk_content_adjusted_correct"] = (
        (out.loc[uk_mask, "exact_match"] == 1)
        | (out.loc[uk_mask, "f1"] >= f1_threshold)
        | (out.loc[uk_mask, "uk_soft_match"] == 1)
    ).astype(int)

    return out


# -----------------------------------------------------------------------------
# Summary and plot
# -----------------------------------------------------------------------------
def build_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for quantization in QUANTIZATION_ORDER:
        qdf = details[details["quantization_name"] == quantization]

        en = qdf[qdf["lang"] == "en"]
        uk = qdf[qdf["lang"] == "uk"]

        if en.empty or uk.empty:
            print(f"[WARN] Missing EN or UK rows for {quantization}")
            continue

        rows.append(
            {
                "quantization_name": quantization,

                "en_n": len(en),
                "uk_n": len(uk),

                "en_f1": round_or_none(en["f1"].mean()),
                "uk_strict_f1": round_or_none(uk["f1"].mean()),
                "uk_content_adjusted_accuracy": round_or_none(
                    uk["uk_content_adjusted_correct"].mean()
                ),

                "uk_strict_em": round_or_none(uk["exact_match"].mean()),
                "en_strict_em": round_or_none(en["exact_match"].mean()),

                "uk_content_gain_vs_strict_f1": round_or_none(
                    uk["uk_content_adjusted_correct"].mean() - uk["f1"].mean()
                ),
                "uk_content_gain_vs_strict_em": round_or_none(
                    uk["uk_content_adjusted_correct"].mean() - uk["exact_match"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_en_f1_vs_uk_content_adjusted(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = summary.copy()
    plot_df["_order"] = plot_df["quantization_name"].map(quantization_sort_key)
    plot_df = plot_df.sort_values("_order")

    plt.figure(figsize=(12, 6))

    plt.plot(
        plot_df["quantization_name"],
        plot_df["en_f1"],
        marker="o",
        label="EN F1",
    )

    plt.plot(
        plot_df["quantization_name"],
        plot_df["uk_strict_f1"],
        marker="o",
        label="UK strict F1",
    )

    plt.plot(
        plot_df["quantization_name"],
        plot_df["uk_content_adjusted_accuracy"],
        marker="o",
        label="UK content-adjusted accuracy",
    )

    plt.ylim(0, 1.0)
    plt.xlabel("Quantization mode")
    plt.ylabel("Score")
    plt.title("QA: English F1 vs Ukrainian strict and content-adjusted scores")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build QA plot with EN F1, UK strict F1, and UK content-adjusted accuracy."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with pilot_qa_details_*_v4.csv files, e.g. results/v4.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <input-dir>/qa_content_accuracy.",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v4",
    )

    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.8,
        help="F1 threshold for Ukrainian content-adjusted correctness.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_dir / "qa_content_accuracy"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== QA UK CONTENT-ADJUSTED ANALYSIS ===")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Version: {args.version}")
    print(f"F1 threshold: {args.f1_threshold}")

    details_raw = read_qa_details(
        input_dir=input_dir,
        version=args.version,
    )

    details = add_uk_content_adjusted_column(
        details=details_raw,
        f1_threshold=args.f1_threshold,
    )

    summary = build_summary(details)

    details_path = output_dir / f"qa_uk_content_adjusted_details_{args.version}.csv"
    summary_path = output_dir / f"qa_en_f1_vs_uk_content_adjusted_{args.version}.csv"
    plot_path = output_dir / f"qa_en_f1_vs_uk_content_adjusted_{args.version}.png"

    # Details are useful for checking which Ukrainian answers were treated as content-correct.
    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {details_path}")
    print(f"[SAVED] {summary_path}")

    plot_en_f1_vs_uk_content_adjusted(
        summary=summary,
        output_path=plot_path,
    )

    print("\n=== QUICK SUMMARY ===")
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()