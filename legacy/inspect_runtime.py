import subprocess
from pathlib import Path

from config import RESULTS_DIR, PRIMARY_MODEL_NAME


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )

        output = []
        output.append(f"$ {' '.join(command)}")
        output.append(f"returncode: {completed.returncode}")

        if completed.stdout:
            output.append("\nSTDOUT:")
            output.append(completed.stdout)

        if completed.stderr:
            output.append("\nSTDERR:")
            output.append(completed.stderr)

        return "\n".join(output)

    except Exception as exc:
        return f"$ {' '.join(command)}\nERROR: {repr(exc)}"


def main():
    out_dir = RESULTS_DIR / "runtime_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = []

    commands = [
        ["ollama", "--version"],
        ["ollama", "list"],
        ["ollama", "show", PRIMARY_MODEL_NAME],
        ["ollama", "ps"],
    ]

    for command in commands:
        sections.append(run_command(command))
        sections.append("\n" + "=" * 80 + "\n")

    report = "\n".join(sections)

    report_path = out_dir / "ollama_runtime_inspection.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved to: {report_path}")


if __name__ == "__main__":
    main()