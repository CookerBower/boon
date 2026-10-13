"""Ability query validation and stable table schemas."""

import json
from typing import cast

import polars as pl
import pytest
from boon import AbilityStat, CalculationError, Demo, StatMode, data, rulesets
from boon.ability_stats import calculate_ability_stats, imbues
from catalog_helpers import write_empty_catalogs


class RecordedResult:
    """Native-method double; cast to Demo only at the public API boundary."""

    def __init__(self):
        self.calls = []

    def _imbues(self, directory, ticks, **kwargs):
        self.calls.append((directory, ticks, kwargs))
        return json.dumps({"bindings": [], "effects": [], "metadata": {}})

    def _calculate_ability_stats(self, directory, ticks, **kwargs):
        self.calls.append((directory, ticks, kwargs))
        return json.dumps(
            {"values": [], "contributions": [], "metadata": {"mode": kwargs["mode"]}}
        )


@pytest.mark.parametrize("function", [imbues, calculate_ability_stats])
@pytest.mark.parametrize(
    "options",
    [
        {"ticks": []},
        {"ticks": -1},
        {"ticks": 2**31 - 1},
        {"ticks": 2**40},
        {"ticks": [True]},
        {"ticks": 1, "steam_ids": [-1]},
        {"ticks": 1, "steam_ids": [0]},
        {"ticks": 1, "steam_ids": [True]},
        {"ticks": 1, "steam_ids": [76561197999389679.0]},
        {"ticks": 1, "steam_ids": [2**64]},
        {"ticks": 1, "data_version": None},
    ],
)
def test_invalid_selection_does_not_download(monkeypatch, function, options):
    monkeypatch.setattr(data, "update", lambda _: pytest.fail("unexpected download"))
    with pytest.raises(ValueError):
        function(RecordedResult(), **({"data_version": "1234"} | options))


@pytest.mark.parametrize(
    "options",
    [
        {"stats": []},
        {"mode": "effective"},
        {"mode": None},
        {"stats": ["unknown"]},
        {"abilities": [-1]},
        {"rulesets": {}},
        {"rulesets": {AbilityStat.RANGE_BONUS: rulesets.range_bonus.v1}},
    ],
)
def test_invalid_stat_query_does_not_download(monkeypatch, options):
    monkeypatch.setattr(data, "update", lambda _: pytest.fail("unexpected download"))
    with pytest.raises(ValueError):
        calculate_ability_stats(
            cast(Demo, RecordedResult()), ticks=50, data_version="1234", **options
        )


@pytest.fixture
def recorded_api(monkeypatch, tmp_path):
    versions = []
    monkeypatch.setattr(
        data, "update", lambda version: versions.append(version) or tmp_path
    )
    return RecordedResult(), versions


@pytest.mark.parametrize(
    ("function", "table", "expected"),
    [
        (imbues, "bindings", {"steam_id": pl.UInt64, "ability_id": pl.UInt32}),
        (imbues, "effects", {"steam_id": pl.UInt64, "value": pl.Float64}),
        (
            calculate_ability_stats,
            "values",
            {"mode": pl.String, "steam_id": pl.UInt64, "value": pl.Float64},
        ),
        (
            calculate_ability_stats,
            "contributions",
            {
                "mode": pl.String,
                "steam_id": pl.UInt64,
                "included": pl.Boolean,
                "source_ability_id": pl.UInt32,
            },
        ),
    ],
    ids=["bindings", "effects", "values", "contributions"],
)
def test_empty_query_tables_keep_their_types(recorded_api, function, table, expected):
    recorded, _ = recorded_api
    result = function(cast(Demo, recorded), ticks=50, data_version="1234")
    frame = getattr(result, table)
    assert {name: frame.schema[name] for name in expected} == expected


@pytest.mark.parametrize("function", [imbues, calculate_ability_stats])
def test_query_requests_the_explicit_catalog_version(recorded_api, function):
    recorded, versions = recorded_api
    function(cast(Demo, recorded), ticks=50, data_version="1234")
    assert versions == ["1234"]


@pytest.mark.parametrize("function", [imbues, calculate_ability_stats])
def test_query_metadata_keeps_the_selected_version(recorded_api, function):
    recorded, _ = recorded_api
    result = function(cast(Demo, recorded), ticks=50, data_version="1234")
    assert result.metadata["data_version"] == "1234"


def test_imbue_selection_reaches_native_query(recorded_api, tmp_path):
    recorded, _ = recorded_api
    imbues(
        cast(Demo, recorded),
        ticks=50,
        steam_ids=[76561197999389679],
        data_version="1234",
    )
    assert recorded.calls == [(tmp_path, [50], {"steam_ids": [76561197999389679]})]


def test_ability_query_forwards_query_flags(recorded_api):
    recorded, _ = recorded_api
    calculate_ability_stats(
        cast(Demo, recorded),
        ticks=[50, 60],
        data_version="1234",
        include_items=True,
        explain=True,
        strict=False,
    )
    options = recorded.calls[0][2]
    assert {name: options[name] for name in ("include_items", "explain", "strict")} == {
        "include_items": True,
        "explain": True,
        "strict": False,
    }


def test_ability_query_defaults_to_current_mode(recorded_api):
    recorded, _ = recorded_api
    result = calculate_ability_stats(
        cast(Demo, recorded), ticks=50, data_version="1234"
    )
    assert recorded.calls[0][2]["mode"] == result.metadata["mode"] == "current"


@pytest.mark.parametrize("function", [imbues, calculate_ability_stats])
def test_native_errors_keep_their_message(monkeypatch, tmp_path, function):
    monkeypatch.setattr(data, "update", lambda _: tmp_path)

    class Failure:
        def _imbues(self, *args, **kwargs):
            raise ValueError("missing replay selection")

        _calculate_ability_stats = _imbues

    with pytest.raises(CalculationError, match="missing replay selection"):
        function(Failure(), ticks=50, data_version="1234")


@pytest.fixture(scope="module")
def native_catalog(tmp_path_factory):
    return write_empty_catalogs(
        tmp_path_factory.mktemp("ability-query-catalog"), "1234"
    )


@pytest.fixture
def installed_native_catalog(monkeypatch, native_catalog):
    monkeypatch.setattr(data, "update", lambda _: native_catalog)


@pytest.fixture(scope="module")
def native_imbues(demo: Demo, native_catalog):
    ticks = demo.ability_ticks["tick"].unique().sort()
    tick = ticks[len(ticks) // 2]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data, "update", lambda _: native_catalog)
        result = demo.imbues(ticks=tick, data_version="1234")
    return tick, result


def test_native_imbue_identity_uses_roster_steam_ids(demo: Demo, native_imbues):
    _, result = native_imbues
    assert "player_slot" not in result.bindings.columns
    assert "player_slot" not in result.effects.columns
    assert result.bindings.height > 0
    assert result.bindings["steam_id"][0] in demo.players["steam_id"]


def test_imbues_have_no_stat_mode(native_imbues):
    assert "mode" not in native_imbues[1].metadata


@pytest.fixture(scope="module")
def selected_native_imbues(demo: Demo, native_imbues, native_catalog):
    tick, full = native_imbues
    steam_id = full.bindings["steam_id"][0]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data, "update", lambda _: native_catalog)
        selected = demo.imbues(ticks=tick, steam_ids=[steam_id], data_version="1234")
    return steam_id, selected


@pytest.mark.parametrize("table", ["bindings", "effects"])
def test_native_imbue_steam_selection_matches_filtered_result(
    native_imbues, selected_native_imbues, table
):
    _, full = native_imbues
    steam_id, selected = selected_native_imbues
    assert getattr(selected, table).equals(
        getattr(full, table).filter(pl.col("steam_id") == steam_id)
    )


@pytest.mark.parametrize("method", ["imbues", "calculate_ability_stats"])
def test_native_ability_query_rejects_unknown_steam_ids(
    demo: Demo, installed_native_catalog, method
):
    tick = demo.ability_ticks["tick"].min()
    with pytest.raises(CalculationError, match="Steam ID 1 has no hero"):
        getattr(demo, method)(ticks=tick, steam_ids=[1], data_version="1234")


def test_empty_native_imbue_selection_returns_no_bindings(
    demo: Demo, installed_native_catalog
):
    tick = demo.ability_ticks["tick"].min()
    assert demo.imbues(
        ticks=tick, steam_ids=[], data_version="1234"
    ).bindings.is_empty()


@pytest.mark.parametrize("mode", list(StatMode))
def test_empty_native_ability_selection_keeps_mode_schema(
    demo: Demo, installed_native_catalog, mode
):
    tick = demo.ability_ticks["tick"].min()
    result = demo.calculate_ability_stats(
        ticks=tick, steam_ids=[], data_version="1234", mode=mode
    )
    assert result.values.is_empty()
    assert "player_slot" not in result.values.columns
    assert "player_slot" not in result.contributions.columns
    assert result.metadata["mode"] == mode
    assert result.values.schema["mode"] == pl.String


@pytest.mark.parametrize(
    "mode", ["baseline", StatMode.BASELINE, "current", StatMode.CURRENT]
)
def test_mode_reaches_native_ability_query(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(data, "update", lambda _: tmp_path)
    demo = RecordedResult()
    result = calculate_ability_stats(
        cast(Demo, demo),
        ticks=50,
        data_version="1234",
        mode=mode,
    )
    assert demo.calls[0][2]["mode"] == mode
    assert result.metadata["mode"] == mode
    assert result.values.schema["mode"] == pl.String
    assert result.contributions.schema["mode"] == pl.String
