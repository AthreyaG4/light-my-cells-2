import numpy as np
import pandas as pd

from lmc.config import ORGANELLE_COLUMNS
from lmc.prep.splits import choose_validation_studies, image_table


def fake_pairs(n_studies=40, seed=0):
    """Studies of uneven size with different organelle sets, plus one two-input study."""
    rng = np.random.default_rng(seed)
    rows = []
    image_id = 0
    for s in range(n_studies):
        organelles = rng.random(4) < 0.6
        organelles[0] = True
        for _ in range(int(rng.integers(3, 60))):
            for modality in (["BF", "PC"] if s == 0 else ["DIC"]):  # study 0 has two inputs per image
                row = {"study": f"Study_{s:02d}", "image_id": image_id, "input_modality": modality}
                for o, has in zip(ORGANELLE_COLUMNS, organelles):
                    row[f"{o}_file"] = f"{o}_{image_id}.tiff" if has else None
                rows.append(row)
            image_id += 1
    return pd.DataFrame(rows)


def per_study(pairs):
    images = image_table(pairs)
    return images.groupby("study")[ORGANELLE_COLUMNS].sum()


def test_two_input_images_count_once():
    pairs = fake_pairs()
    images = image_table(pairs)
    assert not images.duplicated(["study", "image_id"]).any()
    assert len(images) == pairs.drop_duplicates(["study", "image_id"]).shape[0]


def test_same_seed_gives_the_same_choice():
    table = per_study(fake_pairs())
    a, _ = choose_validation_studies(table, 0.10, 0.15, 3, 300, seed=5)
    b, _ = choose_validation_studies(table, 0.10, 0.15, 3, 300, seed=5)
    assert a == b


def test_every_organelle_gets_a_reasonable_share_from_enough_studies():
    table = per_study(fake_pairs())
    chosen, counts = choose_validation_studies(table, 0.10, 0.15, 3, 2000, seed=0)
    total = table.sum().to_numpy()
    share = counts / total
    assert (share <= 0.15 + 1e-9).all()  # no organelle pushed past the cap
    assert (share > 0).all()  # every organelle is represented
    in_val = table.loc[chosen]
    assert ((in_val > 0).sum() >= 3).all()  # at least 3 held-out studies per organelle
