"""Compute the best-focus slice for channels that have files but no best_z in best_focus.tsv.

Run:  python reconcile.py [--force]

Targets are the "must be computed" cases reported by data_validation.py (a blank best_z, or a
channel with no best_z column, where all_files.tsv does list files), e.g. the Study_59 Phase images.
Results go to computed_best_focus.csv (study, image_id, channel, best_z, method), which
data_validation.py and create_pairs.py pick up automatically. best_focus.tsv is never modified.

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

from data_validation import COMPUTED_FILE, DATA_DIR, INPUT_MODALITIES, TO_COMPUTE, check_all, load_files


# The two focus measures below are the ones provided with the challenge (unchanged).
def zfocus_normalized_variance(arr):
    """
    Get the best Zfocus of multipleZ images using normalized variance (recommended for transmitted light images
    Args :
        - An image array of shape TCZYX with T,C= 1
    Out:
        - An image array of best Zfocus with shape ZYX with Z=1
    """
    arr = arr[0,0,...]
    *_, y, x = arr.shape
    mean = arr.mean(axis=(-2, -1))[:, None, None]

    z_plane = ((arr - mean)**2).sum(axis=(-2, -1))[:, None, None] / (y * x * mean)
    return z_plane


def zfocus_auto_correlation(arr):
    """
    Get the best Zfocus of multiple Z stack images using auto correlation, recommended for fluorescence images.
    Args :
        - An image array of shape TCZYX with T,C= 1
    Out:
        - An image array of best Zfocus with shape ZYX with Z=1
    """
    arr = (arr[0,0,...] - arr[0,0,...].mean()) / arr.std()
    auto_corr = (((arr[:, :-1, :] * arr[:, 1:, :]).sum(axis=(-2, -1)))
                 - (arr[:, :-2, :] * arr[:, 2:, :]).sum(axis=(-2, -1)))
    return auto_corr


def best_position(paths, channel):
    """Position (0-based) of the sharpest slice in a stack of files given in z order.

    Transmitted-light channels use normalized variance, organelles use auto correlation;
    the best slice is the one with the highest score.
    """
    stack = np.stack([tifffile.imread(p) for p in paths]).astype(np.float64)[None, None]  # TCZYX, T=C=1
    if channel in INPUT_MODALITIES:
        method, scores = "normalized_variance", zfocus_normalized_variance(stack)
    else:
        method, scores = "auto_correlation", zfocus_auto_correlation(stack)
    scores = scores.ravel()
    if np.isnan(scores).all():
        raise ValueError("every slice scored NaN")
    return int(np.nanargmax(scores)), method


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
