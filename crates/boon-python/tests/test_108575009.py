"""Viewer-verified scoreboard and fight state for match 108575009."""

import gzip
import json
from pathlib import Path

import polars as pl
import pytest
from boon import Demo, data

TICK = 187554
MCGINNIS = 76561198037652386
KELVIN = 76561198055516822
PAIGE = 76561198295494515
SCOREBOARD = json.loads(
    Path(__file__).with_name("108575009-scoreboard.json").read_text()
)


@pytest.fixture(scope="module")
def fight(demo: Demo) -> pl.DataFrame:
    return demo.snapshots(ticks=TICK)


@pytest.fixture(scope="module")
def final_players(demo: Demo) -> pl.DataFrame:
    # Pawns can disappear after death. Retain each player's last recorded row,
    # including the post-game interval, rather than require a pawn at game over.
    return (
        demo.player_ticks.select(
            "tick", "steam_id", "kills", "deaths", "assists", "objective_damage"
        )
        .sort("tick")
        .group_by("steam_id")
        .last()
    )


@pytest.fixture(scope="module")
def summary_frames(demo: Demo):
    return demo.summary()


@pytest.fixture(scope="module")
def final_summary(summary_frames) -> pl.DataFrame:
    return (
        summary_frames["snapshots"].sort("snapshot_time_s").group_by("steam_id").last()
    )


@pytest.fixture(scope="module")
def catalog_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("108575009-catalog")
    source = Path(__file__).parent / "catalogs" / "108575009"
    for name in ("heroes", "abilities", "modifiers", "misc"):
        (directory / f"{name}.json").write_bytes(
            gzip.decompress((source / f"{name}.json.gz").read_bytes())
        )
    return directory


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch, catalog_dir: Path) -> None:
    def installed(version: str) -> Path:
        assert version == "6712"
        return catalog_dir

    monkeypatch.setattr(data, "update", installed)


def assert_displayed_total(actual: int, displayed: str) -> None:
    # The viewer rounds large totals to thousands or tenths of a thousand.
    value = displayed.removesuffix("k")
    precision = 100 if "." in value else 1000
    assert actual == pytest.approx(float(value) * 1000, rel=0, abs=precision / 2)


@pytest.mark.parametrize("expected", SCOREBOARD, ids=lambda row: row["hero"])
def test_final_roster_identity(demo, expected):
    roster = demo.players.filter(pl.col("steam_id") == expected["steam_id"]).row(
        0, named=True
    )
    assert (roster["hero_id"], roster["team_num"]) == (
        expected["hero_id"],
        expected["team_num"],
    )
    if expected["player_name"] is not None:
        assert roster["player_name"] == expected["player_name"]


@pytest.mark.parametrize("expected", SCOREBOARD, ids=lambda row: row["hero"])
@pytest.mark.parametrize("field", ["kills", "deaths", "assists"])
def test_final_live_counter_matches_scoreboard(final_players, expected, field):
    player = final_players.filter(pl.col("steam_id") == expected["steam_id"])
    assert player[field].item() == expected[field]


@pytest.mark.parametrize("expected", SCOREBOARD, ids=lambda row: row["hero"])
@pytest.mark.parametrize(
    "field", ["souls", "player_damage", "objective_damage", "healing_and_barriers"]
)
def test_final_displayed_total_matches_scoreboard(
    final_players, final_summary, expected, field
):
    selection = pl.col("steam_id") == expected["steam_id"]
    if field == "objective_damage":
        actual = final_players.filter(selection)[field].item()
    else:
        row = final_summary.filter(selection).row(0, named=True)
        if field == "healing_and_barriers":
            # The viewer combines these components; the API keeps them separate.
            actual = row["player_healing"] + row["barrier_absorption"]
        else:
            actual = row["net_worth" if field == "souls" else field]
    assert_displayed_total(actual, expected[field])


def test_fight_clock_matches_viewer(demo: Demo) -> None:
    assert demo.tick_to_match_clock(TICK) == "48:21"


def test_fight_snapshot_has_all_player_identities(fight: pl.DataFrame) -> None:
    assert fight.height == fight["steam_id"].n_unique() == 12
    assert "player_slot" not in fight.columns


@pytest.mark.parametrize(
    ("steam_id", "health", "maximum"),
    [(KELVIN, 5039, 5039), (MCGINNIS, 1132, 3837)],
    ids=["Kelvin", "McGinnis"],
)
def test_fight_health_matches_viewer(
    fight: pl.DataFrame, steam_id: int, health: int, maximum: int
) -> None:
    assert fight.filter(pl.col("steam_id") == steam_id).select(
        "health", "max_health"
    ).rows() == [(health, maximum)]


def test_mcginnis_recorded_ammo_fraction(fight: pl.DataFrame) -> None:
    value = fight.filter(pl.col("steam_id") == MCGINNIS)["ammo_fraction"].item()
    assert value == pytest.approx(86 / 89, abs=1e-6)


def test_hidden_king_rejuvenator_holders_match_viewer(
    demo: Demo, fight: pl.DataFrame
) -> None:
    team = demo.players.filter(pl.col("team_num") == 2)["steam_id"]
    holders = fight.filter(
        pl.col("steam_id").is_in(team.implode()) & pl.col("has_rejuvenator")
    )
    assert set(holders["steam_id"]) == set(team) - {76561198074690964}


def test_mcginnis_is_silenced(demo: Demo, catalog: None) -> None:
    states = demo.player_states(ticks=TICK, steam_ids=[MCGINNIS], data_version="6712")
    assert "SILENCED" in states["states"].item().to_list()


def test_mcginnis_fixation_stacks_match_viewer(demo: Demo) -> None:
    # Catalog ID for modifier_passive_haze_stacking_damage, owned by Fixation.
    fixation = (
        demo.active_modifiers.filter(
            (pl.col("tick") <= TICK)
            & (pl.col("hero_id") == 8)
            & (pl.col("modifier_id") == 637809255)
        )
        .sort("tick")
        .group_by("serial")
        .last()
        .filter(pl.col("event") != "removed")
    )
    assert fixation.select("ability_id", "caster_hero_id", "stacks").rows() == [
        (1080948381, 13, 34)
    ]


def test_mcginnis_ammo(demo: Demo, catalog: None) -> None:
    frame = demo.snapshots(ticks=TICK, data_version="6712")
    assert isinstance(frame, pl.DataFrame)
    assert frame.filter(pl.col("steam_id") == MCGINNIS).select(
        "ammo", "max_ammo", "unlimited_ammo"
    ).rows() == [(86, 89, False)]


def test_paige_healing_and_barrier_absorption_stay_separate(
    final_summary: pl.DataFrame,
) -> None:
    assert final_summary.filter(pl.col("steam_id") == PAIGE).select(
        "player_healing", "barrier_absorption"
    ).rows() == [(6081, 21516)]


def test_summary_barrier_absorption_uses_an_unsigned_integer(
    final_summary: pl.DataFrame,
) -> None:
    assert final_summary.schema["barrier_absorption"] == pl.UInt32


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("match_id", 108575009),
        ("build", 10932),
        ("total_ticks", 193317),
        ("winning_team_num", 2),
        ("game_over_tick", 190758),
        ("regulation_ticks", 188897),
        ("regulation_clock_time", "49:11"),
        ("regulation_seconds", pytest.approx(2951.515625)),
    ],
)
def test_match_metadata_matches_viewer(demo: Demo, attribute: str, expected) -> None:
    assert getattr(demo, attribute) == expected


def test_patron_death_marks_game_over(demo: Demo) -> None:
    deaths = demo.objectives.filter(
        (pl.col("objective_type") == "patron") & (pl.col("health") == 0)
    )
    assert deaths["tick"].unique().to_list() == [demo.game_over_tick]


def test_breakable_subclass_counts_match_viewer(demo: Demo) -> None:
    assert demo.breakables.group_by("subclass_id").len().sort("subclass_id").rows() == [
        (202631964, 58),
        (3719077267, 142),
        (3986897915, 355),
    ]


def test_sinners_sacrifice_event_counts_match_viewer(demo: Demo) -> None:
    assert demo.sinners_sacrifice.group_by("event").len().sort("event").rows() == [
        ("hit", 396),
        ("reset", 66),
        ("spawned", 15),
    ]


def test_sinners_sacrifice_total_damage_matches_viewer(demo: Demo) -> None:
    assert demo.sinners_sacrifice["damage"].sum() == 36848


def test_player_melee_damage(demo: Demo) -> None:
    roster = demo.players["hero_id"]
    melee = demo.damage.filter(
        pl.col("is_melee")
        & pl.col("attacker_hero_id").is_in(roster.implode())
        & pl.col("victim_hero_id").is_in(roster.implode())
        & (pl.col("attacker_hero_id") != pl.col("victim_hero_id"))
    )
    assert melee.group_by("melee_type").agg(
        pl.len().alias("hits"), pl.col("damage").sum()
    ).sort("melee_type").rows() == [("heavy", 150, 27420), ("light", 92, 10814)]


def test_healing_intervals_have_positive_length(summary_frames) -> None:
    healing = summary_frames["healing"]
    assert (healing["interval_start_s"] < healing["interval_end_s"]).all()


def test_healing_interval_amounts_are_nonnegative(summary_frames) -> None:
    assert summary_frames["healing"]["amount"].ge(0).all()


def test_healing_keeps_zero_amount_intervals(summary_frames) -> None:
    assert summary_frames["healing"]["amount"].eq(0).any()


@pytest.mark.parametrize(
    ("stat_type", "count", "total"), [("healing", 670, 231435), ("regen", 1312, 224645)]
)
def test_healing_interval_totals_match_recorded_summary(
    summary_frames, stat_type, count, total
):
    rows = summary_frames["healing"].filter(pl.col("stat_type") == stat_type)
    assert (rows.height, rows["amount"].sum()) == (count, total)


@pytest.mark.parametrize("stat_type", ["healing", "regen"])
def test_healing_intervals_agree_with_summary_matrix(summary_frames, stat_type):
    intervals = summary_frames["healing"].filter(pl.col("stat_type") == stat_type)
    recorded = summary_frames["damage"].filter(
        (pl.col("stat_type") == stat_type) & ~pl.col("is_category")
    )
    assert intervals["amount"].sum() == recorded["damage"].sum()


@pytest.mark.parametrize(
    ("column", "self_only"), [("player_healing", False), ("self_healing", True)]
)
def test_cumulative_healing_agrees_with_summary_snapshots(
    summary_frames, column, self_only
):
    healing = summary_frames["healing"]
    compared = 0
    for snapshot in summary_frames["snapshots"].iter_rows(named=True):
        rows = healing.filter(
            (pl.col("interval_end_s") == snapshot["snapshot_time_s"])
            & (pl.col("healer_steam_id") == snapshot["steam_id"])
            & (pl.col("stat_type") == "healing")
        )
        if rows.is_empty():
            continue
        if self_only:
            rows = rows.filter(pl.col("target_steam_id") == snapshot["steam_id"])
        assert rows["total"].sum() == snapshot[column]
        compared += 1
    assert compared > 0


def test_mcginnis_healing_components_match_viewer(final_summary: pl.DataFrame) -> None:
    row = final_summary.filter(pl.col("steam_id") == MCGINNIS).row(0, named=True)
    assert (row["player_healing"], row["self_healing"], row["teammate_healing"]) == (
        18346,
        13853,
        4493,
    )


@pytest.mark.parametrize(
    ("source", "columns", "expected"),
    [
        (
            1,
            ["source_name", "gold", "gold_orbs", "damage"],
            ("k_ePlayers", 7385, 0, 52244),
        ),
        (6, ["gold", "gold_orbs"], (6774, None)),
    ],
)
def test_mcginnis_soul_source_matches_recorded_summary(
    summary_frames, source, columns, expected
):
    rows = summary_frames["gold_sources"].filter(
        (pl.col("steam_id") == MCGINNIS)
        & (pl.col("snapshot_time_s") == 2952)
        & (pl.col("source_id") == source)
    )
    assert rows.select(columns).rows() == [expected]


@pytest.mark.parametrize("dataset", ["snapshots", "last_hits", "gold_sources"])
def test_summary_identity_matches_the_roster(demo: Demo, summary_frames, dataset):
    roster = demo.players.select("steam_id", "hero_id")
    identities = summary_frames[dataset].select("steam_id", "hero_id").unique()
    assert identities["steam_id"].null_count() == 0
    assert identities.join(roster, on=["steam_id", "hero_id"], how="anti").is_empty()


def test_reused_ability_slot_emits_new_identity(demo: Demo) -> None:
    # Slot 3935 is reused with serial 273, a new owner, and a new ability.
    # Its zero cooldown/charge values equal those of the previous occupant.
    rows = demo.ability_ticks.filter(
        (pl.col("tick") == 154504) & (pl.col("ability_id") == 3005970438)
    )
    assert rows.height == 1
