import platform
import sys

import pandas as pd
import psutil

from config import (
    PRIMARY_MODEL_DISPLAY_NAME,
    BACKEND_NAME,
    UA_QA_SUBSET_SIZE,
    EN_QA_SUBSET_SIZE,
    RESULTS_DIR,
)
from dataset_loaders import (
    load_ua_squad_validation_subset,
    load_en_squad_validation_subset,
)


def get_memory_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def main():
    print("=== SMOKE TEST START ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Primary model plan: {PRIMARY_MODEL_DISPLAY_NAME}")
    print(f"Backend plan: {BACKEND_NAME}")
    print(f"Current RAM usage: {get_memory_mb():.2f} MB")

    print("\n[1/3] Loading Ukrainian QA subset...")
    ua_small = load_ua_squad_validation_subset(UA_QA_SUBSET_SIZE)
    print(f"UA subset loaded: {len(ua_small)} examples")

    print("\n[2/3] Loading English QA subset...")
    en_small = load_en_squad_validation_subset(EN_QA_SUBSET_SIZE)
    print(f"EN subset loaded: {len(en_small)} examples")

    print("\n[3/3] Saving preview samples...")
    rows = []

    for lang, ds in [("uk", ua_small), ("en", en_small)]:
        for i in range(min(3, len(ds))):
            ex = ds[i]
            rows.append(
                {
                    "lang": lang,
                    "question": ex["question"],
                    "context_preview": ex["context"][:300].replace("\n", " "),
                    "gold_answers": " | ".join(ex["answers"]["text"]),
                }
            )

    preview_df = pd.DataFrame(rows)
    preview_path = RESULTS_DIR / "smoke_test_dataset_preview.csv"
    preview_df.to_csv(preview_path, index=False, encoding="utf-8-sig")

    print("\nSmoke test passed.")
    print(f"Preview saved to: {preview_path}")
    print(f"Current RAM usage: {get_memory_mb():.2f} MB")
    print("=== SMOKE TEST END ===")


if __name__ == "__main__":
    main()