"""Write input -> organelle pairs (best-focus slices) to a CSV.

Run:  python create_pairs.py [--out input_output_pairs.csv]

One row per (input modality, organelle) for every kept image, ordered by
study, image_id, organelle. input_z / output_z are the z numbers in the filenames
of the best-focus slices (not the TSV's position-style best_z). Images and organelle
channels that data_validation.py says to ignore are left out. Pairs whose best z is
still to be computed (e.g. Study_59 Phase, until reconcile.py has run) are left out
too and counted in the printed summary.
"""
import argparse
from collections import Counter

import pandas as pd

from data_validation import INPUT_MODALITIES, check_all

COLUMNS = ["study", "image_id", "input_modality", "input_z", "input_file", "organelle", "output_z", "output_file"]


def build_pairs():
    _, images = check_all()
    rows, pending = [], Counter()
    for image in images:
        if image["drop_reason"]:
            continue
        inputs = {c: v for c, v in image["channels"].items() if c in INPUT_MODALITIES}
        outputs = {c: v for c, v in image["channels"].items() if c not in INPUT_MODALITIES}
        for modality, (in_z, in_file) in inputs.items():
            for organelle, (out_z, out_file) in outputs.items():
                if in_z is None or out_z is None:
                    pending[image["study"]] += 1
                    continue
                rows.append([image["study"], image["image_id"], modality, in_z, in_file, organelle, out_z, out_file])
    pairs = pd.DataFrame(rows, columns=COLUMNS)
    pairs = pairs.sort_values(["study", "image_id", "organelle", "input_modality"]).reset_index(drop=True)
    return pairs, pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="input_output_pairs.csv")
    args = parser.parse_args()

    pairs, pending = build_pairs()
    pairs.to_csv(args.out, index=False)

    print(f"Wrote {len(pairs)} pairs to {args.out}")
    for organelle, n in pairs.organelle.value_counts().items():
        print(f"  {organelle}: {n}")
    for study, n in pending.items():
        print(f"Left out {n} pairs in {study}: best z still to be computed")


if __name__ == "__main__":
    main()
