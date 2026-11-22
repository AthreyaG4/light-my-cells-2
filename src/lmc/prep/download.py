"""Download only the files listed in input_output_pairs.csv from the Light My Cells share.

Run:  uv run python -m lmc.prep.download [Study_04 Study_27 ...] [--dry-run] [--workers 4] [--out data]

--stacks fetches, before the CSV exists, the full z-stacks of the channels that have files but no
best_z (Study_59 PC), so reconcile.py can score them. --prune-stacks deletes those slices afterwards
except the ones the CSV uses.

The share holds one big zip per study (some are split into .z01, .z02, ... parts). Instead of
downloading them, this reads each zip's file index with HTTP range requests, then fetches and
unpacks just the files we need, each checked against the CRC stored in the zip. Files already
in the output folder are skipped, so re-running resumes. Nothing is deleted except with --prune-stacks.
"""

import argparse
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from lmc.config import DATA_DIR, ORGANELLE_COLUMNS, PAIRS_FILE

SHARE = "https://seafile.lirmm.fr/d/6b9534811c8e4ed989a9"
LISTING = (
    "https://seafile.lirmm.fr/api/v2.1/share-links/6b9534811c8e4ed989a9/dirents/?path=/"
)
ORGANELLES = ORGANELLE_COLUMNS
ZIP64 = 0xFFFFFFFF


def fetch(name, start, end, retries=6):
    """Bytes start..end (inclusive) of a file in the share."""
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{SHARE}/files/",
                params={"p": f"/{name}", "dl": 1},
                headers={"Range": f"bytes={start}-{end}"},
                timeout=120,
            )
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2**attempt)
    raise RuntimeError(f"could not fetch bytes {start}-{end} of {name}")


class Archive:
    """A zip (possibly split into parts) read through range requests, as one virtual file."""

    def __init__(self, parts):
        self.parts = parts  # [(file name in the share, size)] in archive order
        self.start = [0]
        for _, size in parts:
            self.start.append(self.start[-1] + size)
        self.total = self.start[-1]

    def read(self, offset, length):
        length = min(length, self.total - offset)
        out = []
        for i, (name, size) in enumerate(self.parts):
            lo, hi = max(offset, self.start[i]), min(offset + length, self.start[i + 1])
            if lo < hi:
                out.append(fetch(name, lo - self.start[i], hi - self.start[i] - 1))
        return b"".join(out)


def open_archive(study, listing):
    names = [
        f"{study}.z{n:02d}" for n in range(1, 100) if f"{study}.z{n:02d}" in listing
    ]
    names.append(f"{study}.zip")
    return Archive([(n, listing[n]) for n in names])


def read_index(archive):
    """{file name: entry} from the zip's central directory (handles zip64 and split zips)."""
    tail = archive.read(max(0, archive.total - 70000), 70000)
    i = tail.rfind(b"PK\x05\x06")
    _, cd_disk, _, n_total, cd_size, cd_off = struct.unpack(
        "<HHHHII", tail[i + 4 : i + 20]
    )
    if ZIP64 in (cd_size, cd_off) or 0xFFFF in (cd_disk, n_total):
        j = tail.rfind(b"PK\x06\x07", 0, i)  # zip64 end-of-central-directory locator
        z_disk, z_off, _ = struct.unpack("<IQI", tail[j + 4 : j + 20])
        rec = archive.read(archive.start[z_disk] + z_off, 56)
        _, _, _, _, cd_disk, _, n_total, cd_size, cd_off = struct.unpack(
            "<QHHIIQQQQ", rec[4:56]
        )
    cd = archive.read(archive.start[cd_disk] + cd_off, cd_size)

    index, p = {}, 0
    while cd[p : p + 4] == b"PK\x01\x02":
        fields = struct.unpack("<HHHHHHIIIHHHHHII", cd[p + 4 : p + 46])
        (
            _,
            _,
            _,
            method,
            _,
            _,
            crc,
            c_size,
            u_size,
            n_len,
            e_len,
            c_len,
            disk,
            _,
            _,
            off,
        ) = fields
        name = cd[p + 46 : p + 46 + n_len].decode("utf-8", "replace")
        extra = cd[p + 46 + n_len : p + 46 + n_len + e_len]
        q = 0
        while q + 4 <= len(
            extra
        ):  # zip64 extra field replaces the 0xFFFFFFFF placeholders
            tag, size = struct.unpack("<HH", extra[q : q + 4])
            if tag == 1:
                data, k = extra[q + 4 : q + 4 + size], 0
                if u_size == ZIP64:
                    u_size, k = struct.unpack("<Q", data[k : k + 8])[0], k + 8
                if c_size == ZIP64:
                    c_size, k = struct.unpack("<Q", data[k : k + 8])[0], k + 8
                if off == ZIP64:
                    off, k = struct.unpack("<Q", data[k : k + 8])[0], k + 8
                if disk == 0xFFFF:
                    disk = struct.unpack("<I", data[k : k + 4])[0]
            q += 4 + size
        if not name.endswith("/"):
            index[Path(name).name] = {
                "method": method,
                "crc": crc,
                "c_size": c_size,
                "u_size": u_size,
                "n_len": n_len,
                "pos": archive.start[disk] + off,
            }
        p += 46 + n_len + e_len + c_len
    return index


def extract(archive, entry, dest):
    """Fetch one file, check its CRC, and write it to dest."""
    guess = 30 + entry["n_len"] + 256 + entry["c_size"]
    blob = archive.read(entry["pos"], guess)
    n_len, e_len = struct.unpack(
        "<HH", blob[26:30]
    )  # the local header has its own lengths
    begin = 30 + n_len + e_len
    if len(blob) < begin + entry["c_size"]:
        blob += archive.read(
            entry["pos"] + len(blob), begin + entry["c_size"] - len(blob)
        )
    data = blob[begin : begin + entry["c_size"]]
    if entry["method"] == 8:
        data = zlib.decompress(data, -15)
    elif entry["method"] != 0:
        raise ValueError(f"unsupported compression method {entry['method']}")
    if len(data) != entry["u_size"] or zlib.crc32(data) != entry["crc"]:
        raise ValueError("size or CRC mismatch")
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(
        dest
    )  # a file only appears under its real name once it has been verified


def needed_files(pairs, study):
    rows = pairs[pairs["study"] == study]
    names = set(rows["input_file"].dropna())
    for o in ORGANELLES:
        names |= set(rows[f"{o}_file"].dropna())
    return sorted(names)


def stack_files(use_computed=True):
    """{study: slice files} of every channel stack whose best focus has to be computed here.

    These are the channels the TSV has files for but no best_z (see lmc.prep.validate), e.g. the
    Study_59 PC images. With use_computed=True, stacks already in computed_best_focus.csv are left out.
    """
    from lmc.prep.validate import TO_COMPUTE, check_all, load_files

    issues, _ = check_all(use_computed=use_computed)
    tables, wanted = {}, {}
    for t in issues:
        if t["detail"] == TO_COMPUTE:
            table = tables.setdefault(t["study"], load_files(t["study"]))
            stack = table[
                (table.image_id == t["image_id"]) & (table.channel == t["channel"])
            ]
            wanted.setdefault(t["study"], set()).update(stack.filename)
    return {study: sorted(names) for study, names in wanted.items()}


def prune_stacks(out, pairs):
    """Delete the downloaded stack slices that are not pair files. Returns how many were deleted."""
    deleted = 0
    for study, names in stack_files(use_computed=False).items():
        keep = set(needed_files(pairs, study))
        for name in names:
            path = out / study / name
            if name not in keep and path.exists():
                path.unlink()
                deleted += 1
    return deleted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "studies", nargs="*", help="e.g. Study_04 (default: every study in the CSV)"
    )
    parser.add_argument("--out", default=str(DATA_DIR))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--dry-run", action="store_true", help="only report what would be downloaded"
    )
    parser.add_argument(
        "--stacks",
        action="store_true",
        help="fetch the full z-stacks that reconcile.py needs (instead of the files in the CSV)",
    )
    parser.add_argument(
        "--prune-stacks",
        action="store_true",
        help="delete the stack slices fetched by --stacks that are not pair files",
    )
    args = parser.parse_args()

    out = Path(args.out)
    if args.prune_stacks:
        n = prune_stacks(out, pd.read_csv(PAIRS_FILE))
        print(f"Deleted {n} stack slices that are not pair files.")
        return
    if args.stacks:  # needed before input_output_pairs.csv can be built
        wanted = {
            s: n
            for s, n in stack_files().items()
            if not args.studies or s in args.studies
        }
        if not wanted:
            print("No stacks need to be fetched for best-focus computation.")
            return
    else:
        pairs = pd.read_csv(PAIRS_FILE)
        wanted = {
            s: needed_files(pairs, s)
            for s in args.studies or sorted(pairs["study"].unique())
        }
    listing = {
        x["file_name"]: x["size"]
        for x in requests.get(LISTING, timeout=60).json()["dirent_list"]
    }

    total_bytes, problems = 0, []
    for study, names in wanted.items():
        todo = [n for n in names if not (out / study / n).exists()]
        if not todo:
            print(f"{study}: all files already present")
            continue
        archive = open_archive(study, listing)
        index = read_index(archive)
        missing = [n for n in todo if n not in index]
        todo = [n for n in todo if n in index]
        size = sum(index[n]["c_size"] for n in todo)
        total_bytes += size
        print(
            f"{study}: {len(todo)} files to fetch ({size / 1e6:.0f} MB) from {len(archive.parts)} part(s)",
            flush=True,
        )
        problems += [(study, n, "not in the zip") for n in missing]
        if args.dry_run:
            continue

        (out / study).mkdir(parents=True, exist_ok=True)

        def job(name):
            try:
                extract(archive, index[name], out / study / name)
            except Exception as error:
                return (study, name, str(error))

        with ThreadPoolExecutor(args.workers) as pool:
            problems += [r for r in pool.map(job, todo) if r]

    print(
        f"\n{'Would fetch' if args.dry_run else 'Fetched'} {total_bytes / 1e9:.1f} GB (compressed) in total."
    )
    for study, name, why in problems:
        print(f"PROBLEM {study}/{name}: {why}")
    if problems:
        raise SystemExit(f"{len(problems)} problem(s); re-run to retry")


if __name__ == "__main__":
    main()
