"""Default combat preloading must agree with explicit lazy loading."""

from pathlib import Path

import pytest
from boon import Demo
from polars.testing import assert_frame_equal


@pytest.fixture(scope="module")
def loading_modes(demo_paths: list[Path]):
    if not demo_paths:
        pytest.skip("No demo fixtures available")
    pairs = []
    for path in demo_paths:
        eager = Demo(str(path))
        lazy = Demo(str(path), preload=False)
        lazy.load("kills", "damage", "abilities")
        pairs.append((eager, lazy))
    return pairs


@pytest.mark.parametrize("dataset", ["kills", "damage", "abilities"])
def test_default_preload_matches_grouped_lazy_load(loading_modes, dataset):
    for eager, lazy in loading_modes:
        assert_frame_equal(
            getattr(eager, dataset), getattr(lazy, dataset), check_exact=True
        )
