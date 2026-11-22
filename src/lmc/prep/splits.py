"""Assign every image in input_output_pairs.csv to train or validation, holding out whole studies.

Run:  uv run python -m lmc.prep.splits [--val-frac 0.10] [--seed 0] [--out metadata/splits.csv]

Why whole studies: images from one study share cell line, microscope and imaging conditions, and
they look alike (see worklog), so splitting inside a study would leak that into validation and make
scores look better than they are.

Studies differ a lot in size (3 to 644 images) and in which organelles they contain, so the held-out
studies are found by a seeded random search: many random greedy picks, keeping the one where every
organelle's validation share is closest to --val-frac, with at least --min-studies held-out studies
per organelle. The same seed always gives the same split.

One row per (study, image_id); the two inputs of an image (Studies 74-76) share that row, so they
always land on the same side. Writes metadata/splits.csv: study, image_id, split.
"""

import argparse

import numpy as np
import pandas as pd

from lmc.config import ORGANELLE_COLUMNS as ORGANELLES
from lmc.config import PAIRS_FILE, SPLITS_FILE


def image_table(pairs):
    """One row per (study, image_id) with True/False per organelle."""
    files = pairs.groupby(["study", "image_id"])[[f"{o}_file" for o in ORGANELLES]]
    images = files.agg(lambda s: s.notna().any()).reset_index()
    images.columns = ["study", "image_id"] + ORGANELLES
    return images


def choose_validation_studies(per_study, val_frac, max_frac, min_studies, tries, seed):
    """Return (validation studies, images per organelle in them) from a seeded random search."""
    studies = per_study.index.to_numpy()
    counts_per_study = per_study.to_numpy()
    total = counts_per_study.sum(axis=0)
    target, cap = val_frac * total, max_frac * total
    rng = np.random.default_rng(seed)

    best = None
    for _ in range(tries):
        chosen, counts = [], np.zeros(len(ORGANELLES))
        for i in rng.permutation(len(studies)):
            new = counts + counts_per_study[i]
            if (new <= cap).all():  # never let one study push an organelle far past its share
                chosen.append(i)
                counts = new
            if (counts >= target).all():
                break
        enough_studies = (counts_per_study[chosen] > 0).sum(axis=0) >= min_studies
        score = np.abs(counts / total - val_frac).sum()
        key = (not enough_studies.all(), score)  # valid choices first, then closest to the target
        if best is None or key < best[0]:
            best = (key, sorted(studies[chosen]), counts)
    if best[0][0]:
        raise SystemExit(
            f"No choice gave every organelle at least {min_studies} held-out studies "
            f"(try more --tries, a larger --val-frac or a smaller --min-studies)."
        )
    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(PAIRS_FILE))
    parser.add_argument("--out", default=str(SPLITS_FILE))
    parser.add_argument("--val-frac", type=float, default=0.10, help="target share of each organelle's images")
    parser.add_argument("--max-frac", type=float, default=0.15, help="no organelle may exceed this share")
    parser.add_argument("--min-studies", type=int, default=3, help="held-out studies needed per organelle")
    parser.add_argument("--tries", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    images = image_table(pairs)
    per_study = images.groupby("study")[ORGANELLES].sum()
    val_studies, _ = choose_validation_studies(
        per_study, args.val_frac, args.max_frac, args.min_studies, args.tries, args.seed
    )

    images["split"] = np.where(images["study"].isin(val_studies), "validation", "train")
    splits = images[["study", "image_id", "split"]].sort_values(["study", "image_id"])
    assert not splits.duplicated(["study", "image_id"]).any()
    splits.to_csv(args.out, index=False)

    print(f"Wrote {len(splits)} images to {args.out}  (seed {args.seed})")
    print(f"Validation: {len(val_studies)} whole studies, {int((splits.split == 'validation').sum())} images")
    print(f"{'':14}{'train':>7}{'val':>7}{'val share':>11}{'val studies':>13}")
    for o in ORGANELLES:
        has = images[images[o]]
        val = has[has.split == "validation"]
        print(f"{o:14}{len(has) - len(val):>7}{len(val):>7}{len(val) / len(has):>11.1%}{val['study'].nunique():>13}")
    print("Held-out studies:", ", ".join(val_studies))


if __name__ == "__main__":
    main()
