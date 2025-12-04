import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run_step(cmd, title: str, cwd: Path | None = None):
    print("\n" + "=" * 80)
    print(f"STEP: {title}")
    print("COMMAND:", " ".join(str(c) for c in cmd))
    print("=" * 80)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ Step failed: {title}")
        sys.exit(result.returncode)
    print(f"✅ Step completed: {title}")


def main():
    python = sys.executable

    # 1) Build V1 dataset
    run_step(
        [
            python,
            "-m",
            "src.data.build_v1_dataset",
            "--cities",
            "Paris,Lyon,Bordeaux",
            "--min_reviews",
            "3",
        ],
        "Build V1 dataset (listings_v1 & interactions_v1)",
        cwd=PROJECT_ROOT,
    )

    # 2) Precompute embeddings
    run_step(
        [
            python,
            "-m",
            "src.models.precompute_embeddings",
            "--alpha_text",
            "0.4",
            "--max_per_city",
            "1500",      # <= 1500 listings PAR VILLE
            "--max_listings",
            "0",         # <= 0 = pas de limite globale
            "--max_download",
            "1000",
            #"--force",   # on force le recalcul pour appliquer la nouvelle logique
        ],
        "Precompute CLIP embeddings (text + image) and context features",
        cwd=PROJECT_ROOT,
    )


    # 3) Build FAISS index
    run_step(
        [
            python,
            "-c",
            "from src.models.retrieval import build_faiss_index; build_faiss_index()",
        ],
        "Build FAISS index on fused embeddings",
        cwd=PROJECT_ROOT,
    )

    # 4) Lancer API + frontend
    print("\n" + "=" * 80)
    print("Launching API (FastAPI) and UI (Next.js React)...")
    print("API:      http://localhost:8000/docs")
    print("Frontend: http://localhost:3000")
    print("Press Ctrl+C to stop both.")
    print("=" * 80)

    api_cmd = [
        python,
        "-m",
        "uvicorn",
        "src.api.app:app",
        "--reload",
        "--port",
        "8000",
    ]

    # Sous Windows, l’exécutable est souvent npm.cmd
    npm_exe = "npm.cmd" if os.name == "nt" else "npm"
    frontend_cmd = [npm_exe, "run", "dev"]

    api_proc = subprocess.Popen(api_cmd, cwd=PROJECT_ROOT)

    try:
        front_proc = subprocess.Popen(frontend_cmd, cwd=PROJECT_ROOT / "trip-frontend")
    except FileNotFoundError:
        print("❌ Impossible de lancer le frontend : 'npm' non trouvé.")
        print("   → Vérifie que Node.js / npm sont installés et accessibles dans le PATH.")
        print("   → Tu peux aussi aller dans trip-frontend et lancer: npm run dev")
        front_proc = None

    try:
        api_proc.wait()
        if front_proc is not None:
            front_proc.wait()
    except KeyboardInterrupt:
        print("Stopping processes...")
        api_proc.terminate()
        if front_proc is not None:
            front_proc.terminate()


if __name__ == "__main__":
    main()
