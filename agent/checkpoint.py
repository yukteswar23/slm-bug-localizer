"""Experiment checkpoint folders: results/<YYYY-MM-DD>_<NN>_<name>/.

Each folder holds one experiment: config.json, one results file per model,
figures/ (PDF + PNG), and NOTES.md. Full trajectories are too large for the
repo and go to C:/gemma_work/runs/<same folder name>/. Folders are numbered
in creation order; asking for an existing name reuses its folder, so several
models can write into the same experiment.
"""

import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TRAJ_ROOT = Path("C:/gemma_work/runs")
FOLDER = re.compile(r"^\d{4}-\d{2}-\d{2}_(\d{2})_(.+)$")


def checkpoint_dir(name: str, create: bool = True) -> Path:
    existing = {}
    for d in RESULTS.glob("*"):
        if d.is_dir() and (m := FOLDER.match(d.name)):
            existing[m.group(2)] = (int(m.group(1)), d)
    if name in existing:
        return existing[name][1]
    if not create:
        raise FileNotFoundError(f"no checkpoint named {name!r} in {RESULTS}")
    n = max((i for i, _ in existing.values()), default=0) + 1
    d = RESULTS / f"{time.strftime('%Y-%m-%d')}_{n:02d}_{name}"
    (d / "figures").mkdir(parents=True)
    return d


def traj_dir(checkpoint: Path, model: str) -> Path:
    d = TRAJ_ROOT / checkpoint.name / model
    d.mkdir(parents=True, exist_ok=True)
    return d
