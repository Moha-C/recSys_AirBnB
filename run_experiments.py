# run_experiments.py
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(cmd, title: str) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {title}")
    print("COMMAND:", " ".join(str(c) for c in cmd))
    print("=" * 80)
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        print(f"❌ Step failed: {title}")
    else:
        print(f"Step completed: {title}")


def main() -> None:
    # On essaie d'abord le python du venv local, sinon on prend sys.executable
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        python = str(venv_python)
    else:
        python = sys.executable

    # 1) Offline eval : on force l'utilisation des LOGS agrégés (interactions_agg.parquet)
    run_step(
        [
            python,
            "-m",
            "src.eval.offline_eval",
            "--mode",
            "all",
            "--k_eval",
            "10",
            "--use_logs",
            "1",
        ],
        "Offline evaluation (all modes, using LOGS)",
    )

    # 2) Agrégation des logs UI -> interactions_agg.parquet
    run_step(
        [
            python,
            "-m",
            "src.training.aggregate_logs",
        ],
        "Aggregate logs into interactions_agg.parquet",
    )

    # 3) Entraînement du modèle multimodal à partir des logs
    run_step(
        [
            python,
            "-m",
            "src.training.train_multimodal",
            "--use_logs",
            "1",
        ],
        "Train multimodal model from interactions (logs)",
    )


if __name__ == "__main__":
    main()
