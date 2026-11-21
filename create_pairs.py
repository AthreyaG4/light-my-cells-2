"""Write one wide row per input image (with its best-focus organelle images) to a CSV."""

import argparse
from collections import Counter

import pandas as pd

from data_validation import INPUT_MODALITIES, check_all

ORGANELLES = ["Nucleus", "Mitochondria", "Tubulin", "Actin"]
KEY = ["study", "image_id", "input_modality"]
COLUMNS = (
    KEY
    + ["input_z", "input_file"]
    + [f"{o.lower()}_{part}" for o in ORGANELLES for part in ("file", "z")]
)
Z_COLUMNS = ["input_z"] + [f"{o.lower()}_z" for o in ORGANELLES]


def build_rows():
    _, images = check_all()
    rows, pending = [], Counter()
    for image in images:
        if image["drop_reason"]:
            continue
        inputs = {c: v for c, v in image["channels"].items() if c in INPUT_MODALITIES}
        outputs = {
            c: v for c, v in image["channels"].items() if c not in INPUT_MODALITIES
        }
        unknown = set(outputs) - set(ORGANELLES)
        if unknown:
            raise ValueError(
                f"{image['study']} image {image['image_id']}: unknown organelle channel(s) {sorted(unknown)}"
            )
        for modality, (in_z, in_file) in inputs.items():
            row = {
                "study": image["study"],
                "image_id": image["image_id"],
                "input_modality": modality,
                "input_z": in_z,
                "input_file": in_file,
            }
            for organelle in ORGANELLES:
                z, file = outputs.get(organelle, (None, None))
                row[f"{organelle.lower()}_z"], row[f"{organelle.lower()}_file"] = (
                    z,
                    file,
                )
            if in_z is None or all(
                row[f"{o.lower()}_file"] is None for o in ORGANELLES
            ):
                pending[image["study"]] += 1
                continue
            rows.append(row)
    table = pd.DataFrame(rows, columns=COLUMNS).sort_values(KEY).reset_index(drop=True)
    table[Z_COLUMNS] = table[Z_COLUMNS].astype("Int64")
    return table, pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="input_output_pairs.csv")
    args = parser.parse_args()

    table, pending = build_rows()
    table.to_csv(args.out, index=False)

    n_pairs = int(sum(table[f"{o.lower()}_file"].notna().sum() for o in ORGANELLES))
    print(f"Wrote {len(table)} rows ({n_pairs} input-organelle pairs) to {args.out}")
    for organelle in ORGANELLES:
        print(
            f"  {organelle}: {int(table[f'{organelle.lower()}_file'].notna().sum())} rows"
        )
    for study, n in pending.items():
        print(f"Left out {n} rows in {study}: best z still to be computed")


if __name__ == "__main__":
    main()
