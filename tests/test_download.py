"""The range-request zip reader, tested against zips built in memory (no network)."""

import io
import zipfile

import pandas as pd
import pytest

from lmc.prep import download as dl


def make_zip(members):
    """members: {name: (bytes, compression)} -> zip file bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, (data, compression) in members.items():
            z.writestr(zipfile.ZipInfo(name), data, compress_type=compression)
    return buffer.getvalue()


@pytest.fixture
def served(monkeypatch):
    """Serve byte ranges of in-memory 'files' in place of the share; returns the dict to fill."""
    files = {}

    def fake_fetch(name, start, end, retries=6):
        return files[name][start : end + 1]

    monkeypatch.setattr(dl, "fetch", fake_fetch)
    return files


MEMBERS = {
    "Study_X/": (b"", zipfile.ZIP_STORED),
    "Study_X/image_1_DIC_z0.ome.tiff": (bytes(range(256)) * 40, zipfile.ZIP_STORED),
    "Study_X/image_1_Nucleus_z0.ome.tiff": (b"abc" * 5000, zipfile.ZIP_DEFLATED),
    "Study_X/image_2_DIC_z0.ome.tiff": (b"\x00\x01" * 3000, zipfile.ZIP_DEFLATED),
}


def test_index_lists_files_and_skips_folders(served):
    blob = make_zip(MEMBERS)
    served["S.zip"] = blob
    index = dl.read_index(dl.Archive([("S.zip", len(blob))]))
    assert set(index) == {"image_1_DIC_z0.ome.tiff", "image_1_Nucleus_z0.ome.tiff", "image_2_DIC_z0.ome.tiff"}


def test_extract_returns_the_exact_bytes_and_leaves_no_part_file(served, tmp_path):
    blob = make_zip(MEMBERS)
    served["S.zip"] = blob
    archive = dl.Archive([("S.zip", len(blob))])
    index = dl.read_index(archive)
    for name, (data, _) in MEMBERS.items():
        if name.endswith("/"):
            continue
        base = name.split("/")[-1]
        dl.extract(archive, index[base], tmp_path / base)
        assert (tmp_path / base).read_bytes() == data
    assert not list(tmp_path.glob("*.part"))


def test_a_zip_split_into_parts_reads_the_same(served, tmp_path):
    """Parts laid end to end behave as one file, including reads that cross a part boundary."""
    blob = make_zip(MEMBERS)
    cuts = [0, len(blob) // 3, 2 * len(blob) // 3, len(blob)]
    parts = [(f"S.z{i:02d}", blob[a:b]) for i, (a, b) in enumerate(zip(cuts, cuts[1:]), start=1)]
    for name, data in parts:
        served[name] = data
    archive = dl.Archive([(name, len(data)) for name, data in parts])
    index = dl.read_index(archive)
    for name, (data, _) in MEMBERS.items():
        if not name.endswith("/"):
            base = name.split("/")[-1]
            dl.extract(archive, index[base], tmp_path / base)
            assert (tmp_path / base).read_bytes() == data


def test_a_corrupted_file_is_rejected_and_not_written(served, tmp_path):
    blob = bytearray(make_zip(MEMBERS))
    served["S.zip"] = blob
    archive = dl.Archive([("S.zip", len(blob))])
    index = dl.read_index(archive)
    entry = index["image_1_DIC_z0.ome.tiff"]  # stored, so the bytes sit right after the local header
    blob[entry["pos"] + 30 + entry["n_len"] + 5] ^= 0xFF
    with pytest.raises(ValueError, match="CRC"):
        dl.extract(archive, entry, tmp_path / "x.ome.tiff")
    assert not (tmp_path / "x.ome.tiff").exists()
    assert not list(tmp_path.glob("*.part"))


def test_needed_files_collects_inputs_and_organelles_once():
    pairs = pd.DataFrame(
        {
            "study": ["S1", "S1", "S2"],
            "input_file": ["a_BF.tiff", "a_PC.tiff", "b_DIC.tiff"],
            "nucleus_file": ["a_N.tiff", "a_N.tiff", "b_N.tiff"],  # shared by a two-input image
            "mitochondria_file": [None, None, "b_M.tiff"],
            "tubulin_file": [None, None, None],
            "actin_file": [None, None, None],
        }
    )
    assert dl.needed_files(pairs, "S1") == ["a_BF.tiff", "a_N.tiff", "a_PC.tiff"]
    assert dl.needed_files(pairs, "S2") == ["b_DIC.tiff", "b_M.tiff", "b_N.tiff"]
