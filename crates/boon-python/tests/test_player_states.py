"""State query validation, catalog acquisition, and native replay checks."""

import json
from typing import cast

import polars as pl
import pytest
from boon import Demo, data
from boon.player_states import player_states
from catalog_helpers import VERSION, catalog_upstream
from polars.testing import assert_frame_equal


class RecordedResult:
    """Native method double, cast to Demo only at the public boundary."""

    def __init__(self):
        self.calls = []

    def _player_states(self, directory, *, ticks, steam_ids):
        self.calls.append((directory, ticks, steam_ids))
        return pl.DataFrame({"states": [["SPRINTING"]]})


@pytest.fixture
def queried_states(monkeypatch, tmp_path):
    upstream = catalog_upstream(monkeypatch, tmp_path)
    recorded = RecordedResult()
    result = player_states(
        cast(Demo, recorded), ticks=50, steam_ids=[2**63 + 1], data_version=VERSION
    )
    return recorded, result, upstream


def test_player_states_preserve_recorded_labels(queried_states):
    assert queried_states[1].to_dicts() == [{"states": ["SPRINTING"]}]


def test_player_state_selection_preserves_uint64_accounts(queried_states, tmp_path):
    assert queried_states[0].calls == [(tmp_path / VERSION, [50], [2**63 + 1])]


def test_player_state_query_defaults_to_all_ticks_and_accounts(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "update", lambda _: tmp_path)
    recorded = RecordedResult()
    player_states(cast(Demo, recorded), data_version=VERSION)
    assert recorded.calls == [(tmp_path, None, None)]


def test_player_state_query_reuses_verified_catalog_offline(queried_states):
    recorded, _, upstream = queried_states
    before = len(upstream["requests"])
    upstream["offline"] = True
    player_states(cast(Demo, recorded), data_version=VERSION)
    assert len(upstream["requests"]) == before


@pytest.mark.parametrize(
    "options",
    [
        {"ticks": -1},
        {"ticks": True},
        {"ticks": [1.5]},
        {"ticks": [2**31 - 1]},
        {"steam_ids": [0]},
        {"steam_ids": [-1]},
        {"steam_ids": [True]},
        {"steam_ids": [2**64]},
        {"steam_ids": [1.5]},
        {"data_version": ""},
    ],
)
def test_invalid_query_is_rejected_before_download(monkeypatch, options):
    def download(version):
        pytest.fail("invalid input must not download data")

    monkeypatch.setattr(data, "update", download)
    with pytest.raises(ValueError):
        player_states(
            cast(Demo, RecordedResult()), **({"data_version": VERSION} | options)
        )


@pytest.fixture(scope="module")
def state_catalog(tmp_path_factory):
    directory = tmp_path_factory.mktemp("player-state-catalog")
    (directory / "modifiers.json").write_text(
        json.dumps(
            {
                "catalog": "modifiers",
                "client_version": VERSION,
                "source_commit": "a" * 40,
                "modifier_states": {"0": "MODIFIER_STATE_EXAMPLE"},
            }
        )
    )
    return directory


@pytest.fixture(scope="module")
def state_demo(demo_paths):
    if not demo_paths:
        pytest.skip("no demo fixtures")
    return Demo(str(demo_paths[0]), preload=False)


@pytest.fixture
def installed_state_catalog(monkeypatch, state_catalog):
    monkeypatch.setattr(data, "update", lambda _: state_catalog)


@pytest.fixture(scope="module")
def full_states(state_demo, state_catalog):
    steam_id = state_demo.players["steam_id"][0]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data, "update", lambda _: state_catalog)
        result = state_demo.player_states(steam_ids=[steam_id], data_version=VERSION)
    return steam_id, result


@pytest.mark.parametrize(
    "selection", [{"ticks": []}, {"steam_ids": []}], ids=["no-ticks", "no-accounts"]
)
def test_empty_state_selection_keeps_schema(
    state_demo, installed_state_catalog, selection
):
    result = state_demo.player_states(data_version=VERSION, **selection)
    assert result.is_empty()
    assert "player_slot" not in result.columns
    assert result.schema["steam_id"] == pl.UInt64
    assert result.schema["states"] == pl.List(pl.String)
    assert result.schema["unknown_states"] == pl.List(pl.UInt32)


def test_native_state_query_rejects_missing_ticks(state_demo, installed_state_catalog):
    with pytest.raises(ValueError, match="absent from the demo"):
        state_demo.player_states(ticks=2_000_000_000, data_version=VERSION)


def test_full_state_query_keeps_the_selected_account(full_states):
    steam_id, full = full_states
    assert full.height > 0
    assert full["steam_id"].unique().to_list() == [steam_id]


def test_player_states_preserve_unknown_catalog_bits(full_states):
    assert full_states[1]["unknown_states"].list.len().max() > 0


@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_exact_state_queries_match_the_full_pass(
    state_demo, installed_state_catalog, full_states, position
):
    steam_id, full = full_states
    middle = full.height // 2
    indices = {"first": [0], "middle": [middle, middle + 1], "last": [full.height - 1]}[
        position
    ]
    ticks = full["tick"].gather(indices).to_list()
    selected = state_demo.player_states(
        ticks=[*reversed(ticks), ticks[0]], steam_ids=[steam_id], data_version=VERSION
    )
    assert_frame_equal(
        selected, full.filter(pl.col("tick").is_in(ticks)), check_exact=True
    )


def test_old_catalog_gives_actionable_error(demo_paths):
    if not demo_paths:
        pytest.skip("no demo fixtures")
    demo = Demo(str(demo_paths[0]), preload=False)
    with pytest.raises(ValueError, match="boon versions.*boon get"):
        demo.player_states(ticks=1000, data_version=VERSION)
