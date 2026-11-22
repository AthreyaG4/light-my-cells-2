# Light-my-cells-2

Builds input -> organelle image pairs from the Light My Cells study data: a transmitted-light image (BF, DIC, PC) paired with a fluorescence image (Nucleus, Mitochondria, Tubulin, Actin), each taken at its best-focus slice. The dataset is on the Hugging Face Hub (private), with a train and a validation split made of whole studies.

## Layout
```
src/lmc/            the package (installed in editable mode by uv)
  config.py         paths and names used everywhere (ROOT, DATA_DIR, METADATA_DIR, ORGANELLES, ...)
  focus.py          the challenge's two focus functions + best_slice(); shared by data prep and inference
  prep/             data preparation: validate, reconcile, pairs, splits, download, upload
metadata/           small generated files that define the dataset (tracked)
                      input_output_pairs.csv, splits.csv, computed_best_focus.csv
scripts/            run_data_pipeline.sh
tests/              pytest
data/               images and the study-info TSVs (git-ignored)
worklog/            daily notes
```
Training and inference code will go in `src/lmc/` next to `prep/` (e.g. `models/`, `train/`, `infer/`), with configs in `configs/` and outputs in `runs/` (git-ignored).

## Run the data pipeline
Everything in order, stopping at the first failed check (needs bash, e.g. Git Bash; `--dry-run` only checks, `--delete` frees disk as studies are uploaded):
```
bash scripts/run_data_pipeline.sh
```
Or step by step, in this order (`uv run python -m ...`):
```
lmc.prep.validate          # report the data inconsistencies and what to drop/skip
lmc.prep.download --stacks # fetch the full z-stacks that reconcile needs (Study 59 only)
lmc.prep.reconcile         # compute best focus where the TSV has none -> metadata/computed_best_focus.csv
lmc.prep.pairs             # metadata/input_output_pairs.csv (one wide row per input image)
lmc.prep.download          # fetch only the files in the CSV from the share (~30 GB; --dry-run to preview; resumable)
lmc.prep.splits            # metadata/splits.csv: train / validation by whole study (seeded)
lmc.prep.upload            # push to the Hub, a few studies per commit; resumable (set REPO in the file; HF_TOKEN in .env)
```
Tests: `uv run pytest`.

`input_output_pairs.csv` has one row per (study, image_id, input modality) with `<organelle>_file` / `<organelle>_z` columns, empty when that image has no usable image of the organelle. For one organelle's model, keep the rows where its file column is not empty.

## Things to know
- `best_z` in `best_focus.tsv` is a position in the sorted listed slices, not the z number in the filename.
- Study 59 is called `Phase` in its TSV but the files are `PC`; handled in `lmc/prep/validate.py`.
- Every filename in the pairs CSV was checked against the share's zips (none missing).
- Hub shards are named `train-Study_XX.parquet` / `validation-Study_XX.parquet`, so `load_dataset(REPO)` gives `train` and `validation`.

Daily notes are in `worklog/`.
