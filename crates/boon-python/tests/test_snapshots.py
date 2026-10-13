"""Test sampled snapshots from `demo.snapshots(...)`.

The selected rows must equal the same rows from a complete per-tick frame. The
test creates only the selected ticks to use less time and memory. The tests skip
when no `.dem` fixture is present.
"""

import threading

import polars as pl
import pytest
from boon import Demo
from conftest import _require_demo_fixture
from polars.testing import assert_frame_equal


def test_specific_ticks_match_full_frame(demo: Demo) -> None:
    full = demo.player_ticks
    some = sorted(full["tick"].unique().to_list())[100:103]
    snap = demo.snapshots(ticks=some)
    assert isinstance(snap, pl.DataFrame)
    expected = full.filter(pl.col("tick").is_in(some))
    assert snap.sort(["tick", "hero_id"]).equals(expected.sort(["tick", "hero_id"]))


@pytest.fixture(scope="module")
def barrier_reference(demo: Demo):
    full = demo.player_ticks
    populated = full.filter(pl.col("barrier") > 0)["tick"].unique().sort().to_list()
    if not populated:
        pytest.skip("fixture has no recorded barriers")
    ticks = populated[:: max(1, len(populated) // 8)][:8]
    return ticks, full.filter(pl.col("tick").is_in(ticks)).sort(
        ["tick", "steam_id", "hero_id"]
    )


@pytest.fixture(scope="module")
def barrier_readers():
    path = str(_require_demo_fixture())
    return Demo(path, preload=False), Demo(path, preload=False)


@pytest.mark.parametrize("strategy", ["seek", "serial", "segmented"])
def test_barrier_snapshots_match_serial_reference(
    barrier_reference, barrier_readers, monkeypatch, strategy
):
    ticks, expected = barrier_reference
    parsed = barrier_readers[0 if strategy == "seek" else 1]
    if strategy == "seek":
        actual = pl.concat([parsed.snapshots(ticks=tick) for tick in ticks])
    else:
        monkeypatch.setenv("BOON_TICK_SEGMENTS", "1" if strategy == "serial" else "4")
        actual = parsed.snapshots(ticks=ticks, start_tick=ticks[0], end_tick=ticks[-1])
    assert_frame_equal(
        actual.sort(["tick", "steam_id", "hero_id"]), expected, check_exact=True
    )


def test_player_positions_match_snapshot_columns(demo: Demo) -> None:
    ticks = sorted(demo.player_ticks["tick"].unique().to_list())[100:103]
    frame = demo.snapshots(ticks=ticks)
    assert isinstance(frame, pl.DataFrame)
    expected = frame.select("tick", "hero_id", "x", "y")
    positions = demo._player_positions(ticks)
    assert positions.sort(["tick", "hero_id"]).equals(
        expected.sort(["tick", "hero_id"])
    )


def test_single_tick_matches_full_frame(demo: Demo) -> None:
    full = demo.player_ticks
    t = sorted(full["tick"].unique().to_list())[500]
    snap = demo.snapshots(ticks=t)
    assert isinstance(snap, pl.DataFrame)
    expected = full.filter(pl.col("tick") == t)
    assert snap.sort("hero_id").equals(expected.sort("hero_id"))


def test_window_matches_full_frame(demo: Demo) -> None:
    full = demo.player_ticks
    snap = demo.snapshots(start_tick=10000, end_tick=11000)
    assert isinstance(snap, pl.DataFrame)
    expected = full.filter((pl.col("tick") >= 10000) & (pl.col("tick") <= 11000))
    assert snap.sort(["tick", "hero_id"]).equals(expected.sort(["tick", "hero_id"]))


def test_stride_downsamples_to_subset(demo: Demo) -> None:
    full = demo.player_ticks
    snap = demo.snapshots(every=640)
    assert isinstance(snap, pl.DataFrame)
    assert 0 < snap.height < full.height
    # Every sampled row is a real row from the full frame.
    expected = full.filter(pl.col("tick").is_in(snap["tick"].unique().to_list()))
    assert snap.sort(["tick", "hero_id"]).equals(expected.sort(["tick", "hero_id"]))


@pytest.mark.parametrize(
    "events", [["kills"], ["kills", "damage"]], ids=["single-event", "event-union"]
)
def test_event_snapshots_use_only_selected_event_ticks(demo: Demo, events):
    selected = demo.snapshots(events=events)
    event_ticks = {tick for name in events for tick in getattr(demo, name)["tick"]}
    assert set(selected["tick"].unique()) <= event_ticks


def test_message_only_event_ticks_match_loaded_datasets(demo: Demo) -> None:
    events = [
        "kills",
        "damage",
        "flex_slots",
        "abilities",
        "item_purchases",
        "chat",
    ]
    direct_demo = Demo(str(_require_demo_fixture()), preload=False)
    direct = direct_demo.snapshots(events=events)
    assert isinstance(direct, pl.DataFrame)

    ticks = set()
    for name in events:
        ticks.update(getattr(demo, name)["tick"])
    expected = demo.player_ticks.filter(pl.col("tick").is_in(ticks))
    keys = ["tick", "hero_id"]
    assert direct.sort(keys).equals(expected.sort(keys))


def test_single_dataset_returns_frame(demo: Demo) -> None:
    out = demo.snapshots("world_ticks", every=640)
    assert isinstance(out, pl.DataFrame)


def test_multiple_datasets_return_dict(demo: Demo) -> None:
    out = demo.snapshots(["player_ticks", "world_ticks", "troopers"], every=640)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"player_ticks", "world_ticks", "troopers"}


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"every": 64, "seconds": 1.0},
        {"datasets": "not_a_dataset", "every": 64},
        {"every": 0},
    ],
    ids=["no-selector", "conflicting-strides", "unknown-dataset", "zero-stride"],
)
def test_invalid_snapshot_query_is_rejected(demo: Demo, options):
    with pytest.raises(ValueError):
        demo.snapshots(**options)


@pytest.mark.parametrize("cached", [False, True], ids=["unloaded", "cached"])
@pytest.mark.parametrize("dataset", ["not_a_dataset", "healing", "barriers"])
def test_event_selection_rejects_unknown_names_consistently(
    demo: Demo, cached: bool, dataset: str
) -> None:
    parsed = demo if cached else Demo(str(_require_demo_fixture()), preload=False)
    with pytest.raises(ValueError, match="Unknown dataset"):
        parsed.snapshots(events=["kills", dataset])


def test_duplicate_snapshot_names_preserve_return_shape(demo: Demo) -> None:
    expected = demo.snapshots("world_ticks", ticks=1000)
    assert isinstance(expected, pl.DataFrame)
    repeated = demo.snapshots(["world_ticks", "world_ticks"], ticks=1000)
    assert isinstance(repeated, dict)
    assert list(repeated) == ["world_ticks"]
    assert repeated["world_ticks"].equals(expected)


def test_snapshots_release_gil() -> None:
    parsed = Demo(str(_require_demo_fixture()), preload=False)
    ready = threading.Event()
    stop = threading.Event()
    progress = [0]

    def worker() -> None:
        ready.set()
        while not stop.wait(0.001):
            progress[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert ready.wait(timeout=5)
        before = progress[0]
        parsed.snapshots(
            ["player_ticks", "world_ticks", "troopers"],
            every=640,
        )
        after = progress[0]
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert after > before
