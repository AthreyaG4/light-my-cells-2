"""Validate the Light-my-cells study metadata against the 3 inconsistencies we found.

Run:  uv run python -m lmc.prep.validate

How best_z works: `best_z` in best_focus.tsv is a 0-based POSITION in the sorted list of
slices that all_files.tsv lists for that image and channel, not the z number in the filename.
The two are the same unless a stack has gaps or does not start at z0 (Study_24).

The three inconsistencies (per study / image / channel):
  1. BEST_Z_OUT_OF_RANGE  best_z is listed but is not a valid position in the listed slices
  2. BLANK_BEST_Z         best_z is blank in best_focus.tsv
  3. NO_BEST_COLUMN       channel appears in all_files.tsv but best_focus.tsv has no column for it

Rules used to decide what to ignore:
  - image has no valid input (BF/DIC/PC/Phase)  -> drop the image
  - image has no valid output organelle         -> drop the image
  - otherwise only the broken organelle channel is skipped
  - a channel that has files but no best_z (cases 2 and 3) is "to compute" by reconcile.py.
    Once reconcile.py has written computed_best_focus.csv the values are picked up here and
    the channel is treated as valid. Until then it counts as valid for the keep/drop decision
    but has no z yet.
"""
from collections import defaultdict

import pandas as pd

from lmc.config import COMPUTED_FILE, INFO_DIR, INPUT_MODALITIES

CHANNEL_RENAMES = {"Study_59": {"Phase": "PC"}}  # all_files.tsv name -> name used in the files on disk

BEST_Z_OUT_OF_RANGE = "best_z outside the listed slices"
BLANK_BEST_Z = "best_z is blank"
NO_BEST_COLUMN = "channel has no best_z column"
INCONSISTENCIES = [BEST_Z_OUT_OF_RANGE, BLANK_BEST_Z, NO_BEST_COLUMN]

TO_COMPUTE = "files exist, best z must be computed"
RESOLVED = "files exist, best z taken from computed_best_focus.csv"


def study_names():
    return sorted(p.name[: -len("_best_focus.tsv")] for p in INFO_DIR.glob("Study_*_best_focus.tsv"))


def load_files(study):
    """all_files.tsv for a study, with channel names corrected to match the files on disk.

    Study_59's all_files.tsv calls the input channel "Phase" (image_2282_Phase_z0.ome.tiff) but the
    files on disk are named image_2282_PC_z0.ome.tiff, so it is treated as PC everywhere.
    """
    files = pd.read_csv(INFO_DIR / f"{study}_all_files.tsv", sep="\t")
    for old, new in CHANNEL_RENAMES.get(study, {}).items():
        files["channel"] = files.channel.replace(old, new)
        files["filename"] = files.filename.str.replace(f"_{old}_", f"_{new}_", regex=False)
    return files


def load_computed():
    """{(study, image_id, channel): best_z position} from computed_best_focus.csv, empty if absent."""
    if not COMPUTED_FILE.exists():
        return {}
    table = pd.read_csv(COMPUTED_FILE)
    return {(s, int(i), c): int(z) for s, i, c, z in zip(table.study, table.image_id, table.channel, table.best_z)}


def check_study(study, computed):
    """Return (issues, images) for one study.

    issues: list of dicts {study, image_id, channel, kind, detail}
    images: list of dicts {study, image_id, channels, drop_reason, skipped}
        channels: {channel: (z, filename)} for every usable channel, z being the z number in
                  the filename; both are None when the best z still has to be computed.
        drop_reason: None, "no valid input" or "no valid output"
        skipped: channels ignored on this image because of an inconsistency
    """
    best = pd.read_csv(INFO_DIR / f"{study}_best_focus.tsv", sep="\t")
    files = load_files(study)

    filename_of = {(i, c, z): f for i, c, z, f in zip(files.image_id, files.channel, files.z, files.filename)}
    z_numbers = files.groupby(["image_id", "channel"]).z.apply(sorted).to_dict()
    channels_of = files.groupby("image_id").channel.agg(set)
    z_columns = {c[: -len("_best_z")]: c for c in best.columns if c.endswith("_best_z")}

    def pick(image_id, channel, position):
        """(z number, filename) at this position of the listed slices, or None if it is not a valid position."""
        listed = z_numbers.get((image_id, channel), [])
        if not float(position).is_integer() or not 0 <= position < len(listed):
            return None
        z = listed[int(position)]
        return z, filename_of[(image_id, channel, z)]

    issues, images = [], []
    for row in best.to_dict("records"):
        image_id = int(row["image_id"])
        present = channels_of.get(image_id, set())
        usable, skipped = {}, []

        def flag(channel, kind, detail=""):
            issues.append(dict(study=study, image_id=image_id, channel=channel, kind=kind, detail=detail))

        def needs_computing(channel, kind):
            position = computed.get((study, image_id, channel))
            chosen = pick(image_id, channel, position) if position is not None else None
            if chosen:
                flag(channel, kind, RESOLVED)
                usable[channel] = chosen
            else:
                flag(channel, kind, TO_COMPUTE)
                usable[channel] = (None, None)

        for channel, column in z_columns.items():
            position = row[column]
            if pd.isna(position):
                if channel in present:
                    needs_computing(channel, BLANK_BEST_Z)
                else:
                    flag(channel, BLANK_BEST_Z, "no files for this channel")
                    skipped.append(channel)
                continue
            chosen = pick(image_id, channel, position)
            if chosen is None:
                n_listed = len(z_numbers.get((image_id, channel), []))
                flag(channel, BEST_Z_OUT_OF_RANGE, f"best_z={position:g}, but {n_listed} slices listed")
                skipped.append(channel)
            else:
                usable[channel] = chosen

        for channel in sorted(present - set(z_columns)):
            needs_computing(channel, NO_BEST_COLUMN)

        has_input = any(c in INPUT_MODALITIES for c in usable)
        has_output = any(c not in INPUT_MODALITIES for c in usable)
        drop_reason = None if has_input and has_output else ("no valid input" if not has_input else "no valid output")
        images.append(dict(study=study, image_id=image_id, channels=usable, drop_reason=drop_reason, skipped=skipped))
    return issues, images


def check_all(use_computed=True):
    computed = load_computed() if use_computed else {}
    issues, images = [], []
    for study in study_names():
        study_issues, study_images = check_study(study, computed)
        issues += study_issues
        images += study_images
    return issues, images


def _ids(values):
    return ", ".join(str(v) for v in values)


def _print_by_study(rows, show):
    """rows: dicts with study/image_id/channel; show(items) -> text for one study's items."""
    by_study = defaultdict(list)
    for r in rows:
        by_study[r["study"]].append(r)
    for study, items in by_study.items():
        print(f"  {study}:")
        for line in show(items):
            print(f"    {line}")


def main():
    issues, images = check_all()
    print(f"Checked {len(study_names())} studies, {len(images)} images.\n")

    for kind in INCONSISTENCIES:
        found = [i for i in issues if i["kind"] == kind]
        n_images = len({(i["study"], i["image_id"]) for i in found})
        n_studies = len({i["study"] for i in found})
        print(f"[{kind}]  {len(found)} channel cases, {n_images} images, {n_studies} studies")

        def per_image(items):
            channels = defaultdict(list)
            for i in items:
                channels[i["image_id"]].append(i["channel"])
            return [f"image {image_id}: {', '.join(c)}" for image_id, c in channels.items()]

        _print_by_study(found, per_image)
        print()

    dropped = [dict(study=i["study"], reason=i["drop_reason"], image_id=i["image_id"]) for i in images if i["drop_reason"]]
    print(f"IMAGES TO IGNORE (drop entirely): {len(dropped)}")

    def per_reason(items):
        reasons = defaultdict(list)
        for i in items:
            reasons[i["reason"]].append(i["image_id"])
        return [f"{reason}: {_ids(ids)}" for reason, ids in reasons.items()]

    _print_by_study(dropped, per_reason)

    skipped = [dict(study=i["study"], image_id=i["image_id"], channel=c)
               for i in images if not i["drop_reason"] for c in i["skipped"]]
    print(f"\nORGANELLE CHANNELS TO SKIP on kept images: {len(skipped)}")

    def per_channel(items):
        channels = defaultdict(list)
        for i in items:
            channels[i["channel"]].append(i["image_id"])
        return [f"{channel}: {_ids(ids)}" for channel, ids in channels.items()]

    _print_by_study(skipped, per_channel)

    for title, detail in [("BEST Z STILL TO COMPUTE (run lmc.prep.reconcile)", TO_COMPUTE),
                          ("BEST Z RESOLVED from computed_best_focus.csv", RESOLVED)]:
        found = [i for i in issues if i["detail"] == detail]
        print(f"\n{title}: {len(found)} channel cases")

        def per_channel_count(items):
            channels = defaultdict(int)
            for i in items:
                channels[i["channel"]] += 1
            return [f"{channel}: {n} images" for channel, n in channels.items()]

        _print_by_study(found, per_channel_count)


if __name__ == "__main__":
    main()
