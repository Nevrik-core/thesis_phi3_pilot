from config import RESULTS_DIR
from legacy.hf_master_runner import load_master_model


def main():
    out_dir = RESULTS_DIR / "runtime_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, load_duration_sec, rss_mb_after_load, model_dtype = load_master_model()

    report = f"""MASTER MODEL INSPECTION

model_id: microsoft/Phi-4-mini-instruct
model_dtype: {model_dtype}
load_duration_sec: {load_duration_sec:.4f}
rss_mb_after_load: {rss_mb_after_load:.2f}
tokenizer_class: {tokenizer.__class__.__name__}
model_class: {model.__class__.__name__}
"""

    report_path = out_dir / "master_runtime_inspection.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved to: {report_path}")


if __name__ == "__main__":
    main()