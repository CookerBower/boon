"""Urn events verified against current replay records."""

import polars as pl
from boon import Demo
from conftest import _require_demo_fixture


def test_urn_pickup_ticks(demo: Demo) -> None:
    events = demo.urn
    assert events.filter(pl.col("event") == "picked_up")["tick"].to_list() == [
        42869,
        63675,
        80613,
        100510,
        101329,
        120442,
        137871,
        160816,
    ]


def test_urn_completed_delivery_ticks(demo: Demo) -> None:
    events = demo.urn
    assert events.filter(pl.col("event") == "returned")["tick"].to_list() == [
        46108,
        65839,
        82380,
        103965,
        121939,
        139839,
    ]


def test_urn_drop_ticks(demo: Demo) -> None:
    events = demo.urn
    assert events.filter(pl.col("event") == "dropped")["tick"].to_list() == [
        101280,
        166576,
    ]


def test_non_carriers_cannot_complete_urn_deliveries(demo: Demo) -> None:
    events = demo.urn
    assert events.filter(pl.col("tick").is_in([42948, 182712])).is_empty()


def test_urn_only_parse_matches_grouped_load(demo: Demo) -> None:
    events = demo.urn
    standalone = Demo(str(_require_demo_fixture()), preload=False)
    assert standalone.urn.equals(events)
