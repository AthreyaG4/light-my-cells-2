# Light-my-cells-2

Builds input -> organelle image pairs from the Light My Cells study data: a transmitted-light image (BF, DIC, PC) paired with a fluorescence image (Nucleus, Mitochondria, Tubulin, Actin), each taken at its best-focus slice.

Images and study TSVs live in `data/` (git-ignored).

## Run (in this order)
```
uv run python data_validation.py   # report the data inconsistencies and what to drop/skip
uv run python reconcile.py         # compute best focus where the TSV has none -> computed_best_focus.csv
uv run python create_pairs.py      # write input_output_pairs.csv
```

## Things to know
- `best_z` in `best_focus.tsv` is a position in the sorted listed slices, not the z number in the filename.
- Study 59 is called `Phase` in its TSV but the files on disk are `PC`; handled in `data_validation.py`.
- Pair filenames are only checked against disk for studies that are downloaded.

Daily notes are in `worklog/`.
