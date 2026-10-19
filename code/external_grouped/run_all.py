"""Run the three already prepared allocations with resumable checkpoints."""
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
for seed in (42, 43, 44):
    folder = BASE / f"allocation{seed}"
    if not (folder / "PREPARED.json").exists():
        raise RuntimeError(f"Preparation incomplete: {folder}")
    subprocess.run([sys.executable, "-B", str(BASE / "run_reference_models.py"), str(folder)], check=True)
