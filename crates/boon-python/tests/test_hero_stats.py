"""Stat query validation, acquisition, result types, and native errors."""

import json
from typing import cast

import polars as pl
import pytest
from boon import CalculationError, Demo, HeroStat, StatMode, data, rulesets
from boon.hero_stats import calculate_hero_stats
from catalog_helpers import VERSION, catalog_upstream, write_empty_catalogs


class RecordedResult:
    """Native-method double; cast to Demo only at the public API boundary."""

    def __init__(self, **values):
        self.calls = []
        self.values = values

    def _calculate_hero_stats(self, directory, ticks, **kwargs):
        self.calls.append((directory, ticks, kwargs))
        return json.dumps(
            {
                "values": [
                    {
                        "mode": kwargs["mode"],
                        "tick": ticks[0],
                        "steam_id": 76561197999389679,
                        "hero_id": 999,
                        "stat": "clip_size",
                        "value": 36,
                        "unit": "rounds",
                        "ruleset": "clip_size.v1",
                        "status": "calculated",
                        "diagnostic": None,
                        **self.values,
                    }
                ],
                "contributions": [],
                "metadata": {
                    "mode": kwargs["mode"],
                    "source_commit": "a" * 40,
                    "snapshot_version": VERSION,
                },
            }
        )


def test_contributions_retain_target_stats_for_shared_inputs(monkeypatch, tmp_path):
    class Contributions(RecordedResult):
        def _calculate_hero_stats(self, directory, ticks, **kwargs):
            payload = json.loads(
                super()._calculate_hero_stats(directory, ticks, **kwargs)
            )
            payload["contributions"] = [
                {
                    "mode": kwargs["mode"],
                    "tick": ticks[0],
                    "steam_id": 76561197999389679,
                    "hero_id": 999,
                    "stat": stat,
                    "input": "spirit_power",
                    "kind": "flat",
                    "value": 10.0,
                    "source": "test",
                    "definition_path": "/test",
                    "modifier_serial": None,
                }
                for stat in ("clip_size", "weapon_damage")
            ]
            return json.dumps(payload)

    monkeypatch.setattr(data, "update", lambda version: tmp_path)
    report = calculate_hero_stats(
        cast(Demo, Contributions()),
        ticks=50,
        stats=[HeroStat.CLIP_SIZE, HeroStat.WEAPON_DAMAGE],
        data_version=VERSION,
        explain=True,
    )
    assert report.contributions["stat"].to_list() == ["clip_size", "weapon_damage"]
    assert report.contributions["input"].to_list() == ["spirit_power", "spirit_power"]


@pytest.fixture
def downloaded_query(monkeypatch, tmp_path):
    upstream = catalog_upstream(monkeypatch, tmp_path)
    recorded = RecordedResult()
    result = calculate_hero_stats(
        cast(Demo, recorded),
        ticks=50,
        steam_ids=[76561197999389679],
        data_version=VERSION,
        explain=True,
    )
    return recorded, result, upstream


def test_stat_query_preserves_the_native_value_and_identity(downloaded_query):
    _, result, _ = downloaded_query
    assert result.values.select("steam_id", "value").rows() == [
        (76561197999389679, 36.0)
    ]


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ("values", {"steam_id": pl.UInt64, "value": pl.Float64}),
        ("contributions", {"modifier_serial": pl.UInt32, "stat": pl.String}),
    ],
)
def test_stat_query_tables_keep_their_schema(downloaded_query, table, expected):
    frame = getattr(downloaded_query[1], table)
    assert {key: frame.schema[key] for key in expected} == expected


def test_stat_query_metadata_keeps_the_selected_version(downloaded_query):
    assert downloaded_query[1].metadata["data_version"] == VERSION


def test_stat_query_forwards_selection_and_defaults(downloaded_query, tmp_path):
    recorded, _, _ = downloaded_query
    assert recorded.calls == [
        (
            tmp_path / VERSION,
            [50],
            {
                "stats": ["clip_size"],
                "mode": "current",
                "steam_ids": [76561197999389679],
                "heroes": None,
                "explain": True,
                "strict": True,
            },
        )
    ]


def test_stat_query_reuses_verified_catalog_offline(downloaded_query):
    recorded, _, upstream = downloaded_query
    before = len(upstream["requests"])
    upstream["offline"] = True
    calculate_hero_stats(cast(Demo, recorded), ticks=[50, 60], data_version=VERSION)
    assert len(upstream["requests"]) == before


@pytest.mark.parametrize(
    "options",
    [
        {"ticks": []},
        {"ticks": -1},
        {"ticks": 2**31 - 1},
        {"ticks": 2**40},
        {"ticks": 1, "mode": "permanent"},
        {"ticks": 1, "mode": None},
        {"ticks": [True]},
        {"ticks": 1, "stats": []},
        {"ticks": 1, "stats": ["unimplemented_stat"]},
        {"ticks": 1, "steam_ids": [-1]},
        {"ticks": 1, "steam_ids": [0]},
        {"ticks": 1, "steam_ids": [True]},
        {"ticks": 1, "steam_ids": [76561197999389679.0]},
        {"ticks": 1, "rulesets": {}},
        {
            "ticks": 1,
            "rulesets": {
                HeroStat.CLIP_SIZE: rulesets.clip_size.v1,
                HeroStat.BULLET_VELOCITY: rulesets.bullet_velocity.v1,
            },
        },
    ],
)
def test_invalid_queries_fail_before_download(monkeypatch, options):
    def fail(*args, **kwargs):
        pytest.fail("invalid query attempted a download")

    monkeypatch.setattr(data, "update", fail)
    with pytest.raises(ValueError):
        calculate_hero_stats(
            cast(Demo, RecordedResult()), data_version=VERSION, **options
        )


def test_unknown_version_does_not_fall_back():
    with pytest.raises(data.DataError):
        calculate_hero_stats(
            cast(Demo, RecordedResult()), ticks=50, data_version="latest"
        )


def test_ammo_alias():
    assert HeroStat.AMMO is HeroStat.CLIP_SIZE


def test_native_error_is_exposed_as_calculation_error(monkeypatch, tmp_path):
    class Unresolved:
        def _calculate_hero_stats(self, *args, **kwargs):
            raise ValueError("unsupported property scaling")

    monkeypatch.setattr(data, "update", lambda version: tmp_path)
    with pytest.raises(CalculationError, match="unsupported property scaling"):
        calculate_hero_stats(cast(Demo, Unresolved()), ticks=50, data_version=VERSION)


def test_old_catalog_has_actionable_native_error(demo_paths, tmp_path, monkeypatch):
    if not demo_paths:
        pytest.skip("no demo fixtures")
    for name in ("heroes", "abilities", "modifiers", "misc"):
        records = [{"hero_id": 1, "hero_name": "Infernus"}] if name == "heroes" else []
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "catalog": name,
                    "client_version": VERSION,
                    "source_commit": "a" * 40,
                    "records": records,
                }
            )
        )
    monkeypatch.setattr(data, "update", lambda _: tmp_path)
    demo = Demo(str(demo_paths[0]), preload=False)
    with pytest.raises(CalculationError, match="boon versions.*boon get"):
        demo.calculate_hero_stats(ticks=1000, data_version=VERSION)


@pytest.fixture(scope="module")
def query_demo(demo_paths):
    if not demo_paths:
        pytest.skip("no demo fixtures")
    return Demo(str(demo_paths[0]), preload=False)


@pytest.fixture(scope="module")
def native_catalog(tmp_path_factory):
    return write_empty_catalogs(tmp_path_factory.mktemp("hero-query-catalog"), VERSION)


@pytest.fixture
def installed_native_catalog(monkeypatch, native_catalog):
    monkeypatch.setattr(data, "update", lambda _: native_catalog)


def test_native_hero_query_rejects_missing_ticks(query_demo, installed_native_catalog):
    with pytest.raises(CalculationError, match="demo has no tick"):
        query_demo.calculate_hero_stats(ticks=2_000_000_000, data_version=VERSION)


@pytest.mark.parametrize("mode", list(StatMode))
@pytest.mark.parametrize(
    "selection", [{"heroes": []}, {"steam_ids": []}], ids=["no-heroes", "no-accounts"]
)
def test_empty_native_hero_query_keeps_schema_and_mode(
    query_demo, installed_native_catalog, mode, selection
):
    result = query_demo.calculate_hero_stats(
        ticks=1000, mode=mode, data_version=VERSION, stats=list(HeroStat), **selection
    )
    assert result.values.is_empty() and result.contributions.is_empty()
    assert result.values.schema["tick"] == pl.Int32
    assert (
        result.values.schema["mode"] == result.contributions.schema["mode"] == pl.String
    )
    assert result.metadata["mode"] == mode


def test_empty_native_hero_query_reports_the_selected_rules(
    query_demo, installed_native_catalog
):
    result = query_demo.calculate_hero_stats(
        ticks=1000, heroes=[], data_version=VERSION, stats=list(HeroStat)
    )
    assert {rule["id"] for rule in result.metadata["rulesets"]} == {
        f"{stat.value}.v1" for stat in HeroStat
    }


def test_data_version_cannot_silently_select_latest(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("query attempted an implicit latest download")

    monkeypatch.setattr(data, "update", fail)
    with pytest.raises(ValueError, match="explicit client version"):
        calculate_hero_stats(
            cast(Demo, RecordedResult()),
            ticks=50,
            data_version=None,  # ty: ignore[invalid-argument-type] -- Test runtime validation.
        )


@pytest.mark.parametrize(
    "stats",
    [
        [HeroStat.BULLET_VELOCITY],
        list(HeroStat),
        [HeroStat.AMMO, HeroStat.BULLET_VELOCITY, HeroStat.CLIP_SIZE],
    ],
)
@pytest.mark.parametrize("explicit_rules", [False, True])
def test_selected_stats_reach_native_query_once(
    monkeypatch, tmp_path, stats, explicit_rules
):
    monkeypatch.setattr(data, "update", lambda version: tmp_path)
    demo = RecordedResult()
    chosen = {stat: getattr(rulesets, stat.value).v1 for stat in stats}
    calculate_hero_stats(
        cast(Demo, demo),
        ticks=[50, 60],
        data_version=VERSION,
        stats=stats,
        rulesets=chosen if explicit_rules else None,
    )
    assert len(demo.calls) == 1
    assert demo.calls[0][2]["stats"] == list(
        dict.fromkeys(stat.value for stat in stats)
    )


@pytest.mark.parametrize(
    "rule",
    [
        rulesets.clip_size.v1,
        rulesets.Rule("bullet_velocity", "bullet_velocity.v2", 2, "future"),
    ],
    ids=["wrong-stat", "wrong-version"],
)
def test_stat_rejects_wrong_rule_before_download(monkeypatch, rule):
    def fail(version):
        pytest.fail("invalid rules attempted a download")

    monkeypatch.setattr(data, "update", fail)
    with pytest.raises(ValueError, match="select boon.rulesets.bullet_velocity.v1"):
        calculate_hero_stats(
            cast(Demo, RecordedResult()),
            ticks=50,
            data_version=VERSION,
            stats=[HeroStat.BULLET_VELOCITY],
            rulesets={HeroStat.BULLET_VELOCITY: rule},
        )


@pytest.mark.parametrize("strict", [True, False])
def test_partial_values_and_diagnostics_are_retained(monkeypatch, tmp_path, strict):
    monkeypatch.setattr(data, "update", lambda version: tmp_path)
    demo = RecordedResult(
        status="partial", diagnostic="ignored modifiers: unresolved modifier ID 123"
    )
    result = calculate_hero_stats(
        cast(Demo, demo), ticks=50, data_version=VERSION, strict=strict
    )
    assert result.values["value"].item() == 36
    assert result.values["status"].item() == "partial"
    assert "123" in result.values["diagnostic"].item()


@pytest.mark.parametrize(
    ("stat", "unit", "value"),
    [
        (HeroStat.FALLOFF_START, "m", 24.125),
        (HeroStat.FALLOFF_END, "m", 24.125),
        (HeroStat.LIGHT_MELEE_DAMAGE, "damage", 152.012),
        (HeroStat.HEAVY_MELEE_DAMAGE, "damage", 152.012),
        (HeroStat.DEBUFF_RESIST, "%", -8.0),
        (HeroStat.BULLET_LIFESTEAL, "%", 45.4),
        (HeroStat.SPIRIT_LIFESTEAL, "%", 45.4),
        (HeroStat.MELEE_LIFESTEAL, "%", 45.4),
        (HeroStat.BULLET_RESIST, "%", -30.0),
        (HeroStat.SPIRIT_RESIST, "%", -30.0),
        (HeroStat.MELEE_RESIST, "%", -30.0),
        (HeroStat.WEAPON_DAMAGE, "%", -30.0),
    ],
)
def test_native_values_keep_units_precision_and_sign(
    monkeypatch, tmp_path, stat, unit, value
):
    monkeypatch.setattr(data, "update", lambda version: tmp_path)
    demo = RecordedResult(
        stat=stat.value, unit=unit, value=value, ruleset=f"{stat.value}.v1"
    )
    result = calculate_hero_stats(
        cast(Demo, demo), ticks=50, data_version=VERSION, stats=[stat]
    )
    assert result.values.select("stat", "value", "unit", "ruleset").row(0) == (
        stat.value,
        value,
        unit,
        f"{stat.value}.v1",
    )


@pytest.fixture(scope="module")
def recorded_gravity(query_demo, native_catalog):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data, "update", lambda _: native_catalog)
        return query_demo.calculate_hero_stats(
            ticks=10000,
            stats=[HeroStat.GRAVITY_SCALE],
            data_version=VERSION,
            explain=True,
        )


@pytest.fixture(scope="module")
def selected_gravity(query_demo, native_catalog, recorded_gravity):
    steam_id = recorded_gravity.values["steam_id"][0]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data, "update", lambda _: native_catalog)
        result = query_demo.calculate_hero_stats(
            ticks=10000,
            stats=[HeroStat.GRAVITY_SCALE],
            data_version=VERSION,
            steam_ids=[steam_id],
            explain=True,
        )
    return steam_id, result


def test_recorded_gravity_needs_no_hero_or_modifier_definitions(recorded_gravity):
    values = recorded_gravity.values
    assert not values.is_empty()
    assert values["value"].null_count() == 0
    assert values["diagnostic"].null_count() == values.height


def test_gravity_contributions_identify_the_recorded_pawn_field(recorded_gravity):
    values, trace = recorded_gravity.values, recorded_gravity.contributions
    assert trace.height == values.height
    assert set(trace["source"]) == {"replay/player_pawn"}
    assert set(trace["definition_path"]) == {"m_flGravityScale"}
    assert trace["modifier_serial"].null_count() == trace.height
    assert values.sort("steam_id")["value"].equals(trace.sort("steam_id")["value"])


def test_recorded_gravity_identity_matches_roster(query_demo, recorded_gravity):
    assert recorded_gravity.values["steam_id"][0] in query_demo.players["steam_id"]


@pytest.mark.parametrize("table", ["values", "contributions"])
def test_native_hero_steam_selection_matches_filtered_result(
    recorded_gravity, selected_gravity, table
):
    steam_id, selected = selected_gravity
    expected = getattr(recorded_gravity, table).filter(pl.col("steam_id") == steam_id)
    assert getattr(selected, table).equals(expected)


def test_native_hero_query_rejects_unknown_steam_ids(
    query_demo, installed_native_catalog
):
    with pytest.raises(CalculationError, match="Steam ID 1 has no selected hero"):
        query_demo.calculate_hero_stats(
            ticks=10000, data_version=VERSION, steam_ids=[1]
        )


def test_missing_steam_id_keeps_stat_row(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "update", lambda _: tmp_path)
    result = calculate_hero_stats(
        cast(Demo, RecordedResult(steam_id=None)), ticks=50, data_version=VERSION
    )
    assert result.values["steam_id"].item() is None
    assert result.values["value"].item() == 36
    assert "player_slot" not in result.values.columns


def test_old_slot_keyword_is_not_silently_reinterpreted():
    with pytest.raises(TypeError, match="players"):
        calculate_hero_stats(
            cast(Demo, RecordedResult()),
            ticks=50,
            data_version=VERSION,
            players=[0],  # ty: ignore[unknown-argument] -- Reject the old slot keyword.
        )


@pytest.mark.parametrize(
    "mode", ["baseline", StatMode.BASELINE, "current", StatMode.CURRENT]
)
def test_stat_mode_reaches_native_query_and_results(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(data, "update", lambda _: tmp_path)
    demo = RecordedResult()
    result = calculate_hero_stats(
        cast(Demo, demo), ticks=50, data_version=VERSION, mode=mode
    )
    assert demo.calls[0][2]["mode"] == mode
    assert result.values["mode"].item() == mode
    assert result.metadata["mode"] == mode


@pytest.mark.parametrize("strict", [True, False])
def test_strict_query_flag_reaches_native(monkeypatch, tmp_path, strict):
    monkeypatch.setattr(data, "update", lambda _: tmp_path)
    recorded = RecordedResult()
    calculate_hero_stats(
        cast(Demo, recorded), ticks=50, data_version=VERSION, strict=strict
    )
    options = recorded.calls[0][2]
    assert {name: options[name] for name in ("strict", "explain")} == {
        "strict": strict,
        "explain": False,
    }
