"""Compute the best-focus slice for channels that have files but no best_z in best_focus.tsv.

Run:  uv run python -m lmc.prep.reconcile [--force]

Targets are the "must be computed" cases reported by lmc.prep.validate (a blank best_z, or a
channel with no best_z column, where all_files.tsv does list files), e.g. the Study_59 Phase images.
Results go to computed_best_focus.csv (study, image_id, channel, best_z, method), which
lmc.prep.validate and lmc.prep.pairs pick up automatically. best_focus.tsv is never modified.

best_z written here follows the same convention as the TSV: a 0-based position in the sorted
list of slices that all_files.tsv lists for that image and channel.

Targets whose image files are not on disk are skipped and reported. Existing results are kept
unless --force is given.
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import tifffile

from lmc.config import COMPUTED_FILE, DATA_DIR
from lmc.focus import best_slice
from lmc.prep.validate import TO_COMPUTE, check_all, load_files


def best_position(paths, channel):
    """Position (0-based) of the sharpest slice in a stack of files given in z order."""
    stack = np.stack([tifffile.imread(p) for p in paths])
    return best_slice(stack, channel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute targets that already have a result")
    args = parser.parse_args()

    # Targets: judged without the existing results, so --force and re-runs see the same list.
    issues, _ = check_all(use_computed=False)
    targets = [i for i in issues if i["detail"] == TO_COMPUTE]
    results = {}  # (study, image_id, channel) -> (best_z, method)
    if COMPUTED_FILE.exists():
        table = pd.read_csv(COMPUTED_FILE)
        for r in table.itertuples():
            results[r.study, int(r.image_id), r.channel] = (int(r.best_z), r.method)

    computed_now, not_on_disk, failed, example = 0, Counter(), [], {}
    files_of = {}
    for t in targets:
        key = (t["study"], t["image_id"], t["channel"])
        if key in results and not args.force:
            continue
        if t["study"] not in files_of:
            files_of[t["study"]] = load_files(t["study"])
        table = files_of[t["study"]]
        stack = table[(table.image_id == t["image_id"]) & (table.channel == t["channel"])].sort_values("z")
        paths = [DATA_DIR / t["study"] / name for name in stack.filename]
        missing = [p for p in paths if not p.exists()]
        if missing:
            not_on_disk[t["study"], t["channel"]] += 1
            example[t["study"], t["channel"]] = missing[0].name
            continue
        try:
            position, method = best_position(paths, t["channel"])
        except Exception as error:  # a bad file should not stop the whole run
            failed.append((*key, str(error)))
            continue
        results[key] = (position, method)
        computed_now += 1

    if computed_now:
        rows = [[*key, position, method] for key, (position, method) in sorted(results.items())]
        pd.DataFrame(rows, columns=["study", "image_id", "channel", "best_z", "method"]).to_csv(COMPUTED_FILE, index=False)

    print(f"Targets: {len(targets)} channel cases that need a computed best z.")
    print(f"Computed {computed_now} this run; {len(results)} in {COMPUTED_FILE.name}.")
    for (study, channel), n in not_on_disk.items():
        print(f"Skipped {n} {channel} images in {study}: files not on disk, e.g. {example[study, channel]} "
              f"(download the study or check the filenames, then re-run).")
    for study, image_id, channel, error in failed:
        print(f"Failed {study} image {image_id} {channel}: {error}")


if __name__ == "__main__":
    main()
