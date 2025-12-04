# run_experiments.py
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(cmd, title):
    print("\n" + "=" * 80)
    print(f"STEP: {title}")
    print("COMMAND:", " ".join(str(c) for c in cmd))
    print("=" * 80)
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        print(f"❌ Step failed: {title}")
        sys.exit(r.returncode)
    print(f"✅ Step completed: {title}")


def main():
    python = sys.executable

    # Offline eval
    run_step(
        [
            python,
            "-m",
            "src.eval.offline_eval",
            "--mode",
            "all",
            "--k_eval",
            "10",
        ],
        "Offline evaluation (all modes)",
    )

    # Aggregate logs
# Aggregate logs
    run_step(
        [
            python,
            "-m",
            "src.training.aggregate_logs",
        ],
        "Aggregate logs into interactions_from_logs.parquet",
    )

    # Train multimodal model
    run_step(
        [
            python,
            "-m",
            "src.training.train_multimodal",
            "--use_logs",
            "1",
        ],
        "Train multimodal model from interactions (reviews + logs)",
    )


if __name__ == "__main__":
    main()
