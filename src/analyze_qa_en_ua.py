from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def quantization_sort_key(q: str) -> int:
    order = {
        "BF16": 0,
        "Q8_0": 1,
        "Q6_K": 2,
        "Q5_K_M": 3,
        "Q4_K_M": 4,
    }
    return order.get(str(q).upper(), 999)


def load_qa_content_accuracy_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Якщо раптом у тебе інші назви колонок, тут можна додати мапінг
    rename_map = {
        "exact_match": "strict_exact_match",
        "em": "strict_exact_match",
        "f1": "avg_f1",
        "content_accuracy": "content_accuracy_lenient",
        "content_aware_accuracy": "content_accuracy_lenient",
    }
    df = df.rename(columns=rename_map)

    required_columns = {
        "lang",
        "quantization_name",
        "strict_exact_match",
        "avg_f1",
        "content_accuracy_lenient",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["lang"] = df["lang"].astype(str).str.lower().str.strip()
    df["quantization_name"] = df["quantization_name"].astype(str).str.strip()

    return df


def plot_strict_vs_content_uk_en_single(table: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))

    styles = {
        "uk": "-",
        "en": "--",
    }

    for lang in ["uk", "en"]:
        d = table[table["lang"] == lang].copy()

        if d.empty:
            print(f"[WARN] No rows found for lang={lang}")
            continue

        d["_order"] = d["quantization_name"].map(quantization_sort_key)
        d = d.sort_values("_order")

        ls = styles[lang]

        plt.plot(
            d["quantization_name"],
            d["strict_exact_match"],
            marker="o",
            linestyle=ls,
            label=f"{lang.upper()} Strict EM",
        )
        plt.plot(
            d["quantization_name"],
            d["avg_f1"],
            marker="o",
            linestyle=ls,
            label=f"{lang.upper()} Average F1",
        )
        plt.plot(
            d["quantization_name"],
            d["content_accuracy_lenient"],
            marker="o",
            linestyle=ls,
            label=f"{lang.upper()} Content-aware adjusted accuracy",
        )

    plt.ylim(0, 1.0)
    plt.xlabel("Quantization mode")
    plt.ylabel("Score")
    plt.title("QA: Ukrainian and English strict metrics vs content-aware adjusted accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {output_path}")


def main() -> None:
    # Вкажи свій CSV, де вже є і uk, і en
    input_csv = Path(r"C:\Users\ilina\thesis_phi3_pilot\results\v4\qa_content_accuracy_both_langs_v4.csv")
    output_png = Path(r"C:\Users\ilina\thesis_phi3_pilot\results\v4\qa_content_accuracy_uk_en_v4.png")

    df = load_qa_content_accuracy_table(input_csv)

    print("Loaded rows:", len(df))
    print(df[["lang", "quantization_name", "strict_exact_match", "avg_f1", "content_accuracy_lenient"]])

    plot_strict_vs_content_uk_en_single(df, output_png)


if __name__ == "__main__":
    main()