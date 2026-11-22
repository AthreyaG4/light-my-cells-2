"""Build the wide dataset study by study and push it to the Hugging Face Hub (private).

Set REPO below and put HF_TOKEN=<your token> in .env (git-ignored), then run:
    uv run python -m lmc.prep.upload [--max-gb 3] [--delete]

One parquet shard per study, named after its split from splits.csv (data/train-Study_XX.parquet or
data/validation-Study_XX.parquet in the repo, so the Hub shows a train and a validation split;
run lmc.prep.splits first). Shards are uploaded a few studies per
commit (--max-gb of image files each), so only one batch of working files is on disk at a time.
Studies already in the repo are skipped, so a re-run resumes. --delete removes a study's image
files from disk once its batch is uploaded.
One row per input image; each organelle column is None when that image has no image of it.
Only studies whose folder exists in data/ are included.
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset, Features, Value
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from lmc.config import DATA_DIR, ENV_FILE, ORGANELLE_COLUMNS, PAIRS_FILE, SPLITS_FILE

REPO = "AG2307/light-my-cells"
DATA = DATA_DIR
SPLIT_NAMES = ["train", "validation"]
load_dotenv(ENV_FILE)
ORGANELLES = ORGANELLE_COLUMNS

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


def study_rows(study):
    pairs = pd.read_csv(PAIRS_FILE)
    pairs = pairs[pairs["study"] == study]
    return pairs.astype(object).where(pairs.notna(), None).to_dict("records")


def files_of(study):
    """Every image file the rows of this study point to (error if one is not on disk)."""
    names = set()
    for r in study_rows(study):
        names |= {
            r[c] for c in ["input_file"] + [f"{o}_file" for o in ORGANELLES] if r[c]
        }
    paths = [DATA / study / n for n in sorted(names)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{study}: {len(missing)} files missing, e.g. {missing[0]} (run `uv run python -m lmc.prep.download`)"
        )
    return paths


def read_bytes(study, filename):
    return (DATA / study / filename).read_bytes() if isinstance(filename, str) else None


def rows(study):
    for r in study_rows(study):
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


def study_splits():
    """{study: 'train' or 'validation'} from splits.csv (whole studies are held out, see lmc/prep/splits.py)."""
    if not SPLITS_FILE.exists():
        raise SystemExit(f"{SPLITS_FILE.name} not found: run `uv run python -m lmc.prep.splits` first")
    splits = pd.read_csv(SPLITS_FILE)
    per_study = splits.groupby("study")["split"].agg(set)
    mixed = [s for s, v in per_study.items() if len(v) != 1]
    if mixed:
        raise SystemExit(f"{SPLITS_FILE.name} puts images of {mixed[0]} in both splits; splits must be per study")
    result = {s: next(iter(v)) for s, v in per_study.items()}
    missing = set(pd.read_csv(PAIRS_FILE)["study"]) - set(result)
    if missing:
        raise SystemExit(f"{SPLITS_FILE.name} has no split for {sorted(missing)[0]}: re-run `uv run python -m lmc.prep.splits`")
    return result


def build_shard(study, split, out_dir, cache_dir):
    ds = Dataset.from_generator(
        rows, features=FEATURES, gen_kwargs={"study": study}, cache_dir=str(cache_dir)
    )
    ds.to_parquet(out_dir / f"{split}-{study}.parquet")  # the file name makes it a train or validation shard on the Hub
    del ds  # release the memory-mapped cache so it can be deleted on Windows
    shutil.rmtree(cache_dir, ignore_errors=True)


def studies_in_repo(api):
    """{study: split} for the shards already in the Hub repo (empty if the repo does not exist yet)."""
    try:
        files = api.list_repo_files(REPO, repo_type="dataset")
    except RepositoryNotFoundError:
        return {}
    found = {}
    for name in (Path(f).name for f in files):
        for split in SPLIT_NAMES:
            if name.startswith(f"{split}-") and name.endswith(".parquet"):
                found[name[len(split) + 1 : -len(".parquet")]] = split
    return found


def upload(api, folder, message, tries=5):
    for attempt in range(tries):
        try:
            return api.upload_folder(
                repo_id=REPO,
                repo_type="dataset",
                folder_path=folder,
                path_in_repo="data",
                commit_message=message,
            )
        except Exception as error:  # e.g. a 429 rate limit: wait and try again
            if attempt == tries - 1:
                raise
            print(
                f"upload failed ({error.__class__.__name__}); retrying in {60 * (attempt + 1)} s"
            )
            time.sleep(60 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-gb", type=float, default=3.0, help="image data per upload batch"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="delete a study's image files after it is uploaded",
    )
    args = parser.parse_args()

    splits = study_splits()
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    in_repo = studies_in_repo(api)
    wrong = [s for s, split in in_repo.items() if s in splits and splits[s] != split]
    if wrong:  # uploading again would leave the study in the repo twice
        raise SystemExit(
            f"{wrong[0]} is in the Hub repo as '{in_repo[wrong[0]]}' but {SPLITS_FILE.name} now says "
            f"'{splits[wrong[0]]}'. Restore the old splits.csv, or delete that shard from the repo first."
        )

    pairs = pd.read_csv(PAIRS_FILE)
    studies = [
        s
        for s in sorted(pairs["study"].unique())
        if (DATA / s).is_dir() and s not in in_repo
    ]
    print(
        f"{len(studies)} studies to upload (skipping {len(set(pairs['study'])) - len(studies)})"
    )

    batches, batch, batch_bytes = [], [], 0
    for s in studies:
        size = sum(p.stat().st_size for p in files_of(s))
        if batch and batch_bytes + size > args.max_gb * 1e9:
            batches.append(batch)
            batch, batch_bytes = [], 0
        batch.append(s)
        batch_bytes += size
    if batch:
        batches.append(batch)

    for batch in batches:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            out = Path(tmp) / "data"
            out.mkdir()
            for s in batch:
                build_shard(s, splits[s], out, Path(tmp) / "cache")
            upload(
                api,
                out,
                f"Add {batch[0]} to {batch[-1]}"
                if len(batch) > 1
                else f"Add {batch[0]}",
            )
        print(f"uploaded {', '.join(batch)}", flush=True)
        if args.delete:
            for s in batch:
                for p in files_of(s):
                    p.unlink()
                if not any((DATA / s).iterdir()):
                    (DATA / s).rmdir()


if __name__ == "__main__":
    main()
