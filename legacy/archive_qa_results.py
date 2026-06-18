from shutil import copy2

from config import RESULTS_DIR


ARCHIVE_DIR = RESULTS_DIR / "qa_squad_100_baseline"

FILES_TO_ARCHIVE = [
    "pilot_qa_details_100.csv",
    "pilot_qa_summary_100.csv",
    "pilot_qa_details_master_100.csv",
    "pilot_qa_summary_master_100.csv",

]


def main():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    for filename in FILES_TO_ARCHIVE:
        src = RESULTS_DIR / filename

        if not src.exists():
            missing.append(filename)
            continue

        dst = ARCHIVE_DIR / filename

        if dst.exists():
            print(f"SKIP already exists: {dst}")
            continue

        copy2(src, dst)
        copied.append(filename)
        print(f"COPIED: {src} -> {dst}")

    print("\n=== ARCHIVE SUMMARY ===")
    print(f"Archive dir: {ARCHIVE_DIR}")
    print(f"Copied: {copied}")
    print(f"Missing: {missing}")


if __name__ == "__main__":
    main()