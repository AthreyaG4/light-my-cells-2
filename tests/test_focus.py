import numpy as np
import pytest

from lmc.focus import best_slice


def test_transmitted_light_picks_the_high_contrast_slice():
    rng = np.random.default_rng(0)
    stack = np.full((5, 32, 32), 100.0) + rng.normal(0, 0.5, (5, 32, 32))  # flat, blurry slices
    stack[3] = 100.0 + rng.normal(0, 20, (32, 32))  # sharp one
    position, method = best_slice(stack, "DIC")
    assert position == 3
    assert method == "normalized_variance"


def test_fluorescence_picks_the_structured_slice():
    rng = np.random.default_rng(1)
    stack = rng.normal(0, 1, (6, 32, 32))  # noise has ~no spatial correlation
    stack[2, 8:24, 8:24] += 10.0  # a coherent bright object
    position, method = best_slice(stack, "Nucleus")
    assert position == 2
    assert method == "auto_correlation"


@pytest.mark.parametrize("channel", ["BF", "DIC", "PC", "Phase"])
def test_every_input_modality_uses_normalized_variance(channel):
    stack = np.random.default_rng(2).normal(50, 5, (3, 16, 16))
    assert best_slice(stack, channel)[1] == "normalized_variance"


@pytest.mark.parametrize("channel", ["Nucleus", "Mitochondria", "Tubulin", "Actin"])
def test_every_organelle_uses_auto_correlation(channel):
    stack = np.random.default_rng(3).normal(50, 5, (3, 16, 16))
    assert best_slice(stack, channel)[1] == "auto_correlation"


@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_a_constant_fluorescence_stack_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError):
        best_slice(np.zeros((3, 8, 8)), "Nucleus")
