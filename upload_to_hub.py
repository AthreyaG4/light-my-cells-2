"""Build the wide dataset from input_output_pairs.csv and push it to the Hugging Face Hub (private).

Set REPO below and put HF_TOKEN=<your token> in .env (git-ignored), then run:  python upload_to_hub.py
One row per input image; each organelle column is None when that image has no image of it.
Only studies whose folder exists in data/ are included.
"""

from pathlib import Path

import pandas as pd
from datasets import Dataset, Features, Value
from dotenv import load_dotenv

REPO = "AG2307/light-my-cells"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")
ORGANELLES = ["nucleus", "mitochondria", "tubulin", "actin"]

feats = {
    "study": Value("string"),
    "scene_id": Value("int64"),
    "modality": Value("string"),
    "input": Value("binary"),
    "input_z": Value("int32"),
}
for o in ORGANELLES:
    feats[o] = Value("binary")
    feats[f"{o}_z"] = Value("int32")
FEATURES = Features(feats)


def read_bytes(study, filename):
    return (DATA / study / filename).read_bytes() if isinstance(filename, str) else None


def rows():
    pairs = pd.read_csv(ROOT / "input_output_pairs.csv")
    pairs = pairs[pairs["study"].isin([p.name for p in DATA.glob("Study_*")])]
    for r in pairs.astype(object).where(pairs.notna(), None).to_dict("records"):
        row = {
            "study": r["study"],
            "scene_id": int(r["image_id"]),
            "modality": r["input_modality"],
            "input": read_bytes(r["study"], r["input_file"]),
            "input_z": int(r["input_z"]),
        }
        for o in ORGANELLES:
            row[o] = read_bytes(r["study"], r[f"{o}_file"])
            row[f"{o}_z"] = None if r[f"{o}_z"] is None else int(r[f"{o}_z"])
        yield row


ds = Dataset.from_generator(rows, features=FEATURES)
print(ds)
ds.push_to_hub(REPO, private=True)
