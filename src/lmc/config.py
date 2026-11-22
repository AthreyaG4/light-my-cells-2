"""Where things live and the names used everywhere. Other modules import these, nothing is repeated.

ROOT is found from this file's location (src/lmc/config.py), so the package must be installed
in editable mode (what `uv sync` / `uv run` do for this project).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"  # images and the study-info TSVs (git-ignored)
INFO_DIR = DATA_DIR / "Study_info_filenames_and_optimal focus"
METADATA_DIR = ROOT / "metadata"  # small generated files that define the dataset (tracked)
PAIRS_FILE = METADATA_DIR / "input_output_pairs.csv"
SPLITS_FILE = METADATA_DIR / "splits.csv"
COMPUTED_FILE = METADATA_DIR / "computed_best_focus.csv"  # written by reconcile: study,image_id,channel,best_z,method
ENV_FILE = ROOT / ".env"  # HF_TOKEN lives here (git-ignored)

INPUT_MODALITIES = {"BF", "DIC", "PC", "Phase"}  # transmitted light; everything else is an organelle
ORGANELLES = ["Nucleus", "Mitochondria", "Tubulin", "Actin"]  # channel names in the TSVs and filenames
ORGANELLE_COLUMNS = [o.lower() for o in ORGANELLES]  # column prefixes in the pairs CSV and the Hub dataset
