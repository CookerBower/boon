"""Catalog-backed names, automatic acquisition, and version selection."""

import json
import urllib.error
from pathlib import Path

import pytest
from boon import (
    Demo,
    ability_display_names,
    ability_names,
    breakable_names,
    data,
    hero_names,
    modifier_names,
)
from catalog_helpers import VERSION, catalog_upstream, release


@pytest.fixture
def upstream(monkeypatch, tmp_path):
    records = json.loads(Path(__file__).with_name("name-catalogs.json").read_text())
    return catalog_upstream(monkeypatch, tmp_path / "cache", records)


@pytest.mark.parametrize(
    "lookup",
    [hero_names, ability_names, ability_display_names, modifier_names, breakable_names],
)
def test_name_lookup_downloads_once_and_reuses_catalog_offline(upstream, lookup):
    assert lookup()
    assert data.local_versions()[0]["client_version"] == VERSION
    before = len(upstream["requests"])
    upstream["offline"] = True
    assert lookup()
    assert len(upstream["requests"]) == before


@pytest.mark.parametrize(
    ("lookup", "key", "expected"),
    [
        (hero_names, 1, "Infernus"),
        (ability_names, 46922526, "inherent_base"),
        (ability_display_names, "citadel_ability_hook", "Grapple Arm"),
        (modifier_names, 1364211883, "ability_afterburn/modifier_afterburn_dot"),
        (modifier_names, 3285944580, "modifier_afterburn_dot"),
        (breakable_names, 3986897915, "citadel_breakable_prop_wooden_crate"),
    ],
)
def test_catalog_names_match_the_selected_record(upstream, lookup, key, expected):
    assert lookup()[key] == expected


def test_modifying_returned_names_does_not_change_the_cache(upstream):
    names = hero_names()
    names[1] = "caller changed this"
    assert hero_names()[1] == "Infernus"


@pytest.fixture(
    params=[
        pytest.param((hero_names, "heroes", "hero_id", "display_name"), id="heroes"),
        pytest.param(
            (breakable_names, "misc", "misc_id", "misc_name"), id="breakables"
        ),
    ]
)
def lookup_source(request):
    return request.param


def replace_name(upstream, lookup_source, label, version=VERSION):
    _, catalog, id_key, label_key = lookup_source
    record = upstream["records"][catalog][0]
    record[label_key] = label
    upstream["index"], upstream["files"] = release(version, records=upstream["records"])
    return record[id_key]


def test_default_names_do_not_refresh_the_remote_index(upstream, lookup_source):
    lookup = lookup_source[0]
    lookup()
    changed_id = replace_name(upstream, lookup_source, "Changed name", "6701")
    before = len(upstream["requests"])
    assert lookup()[changed_id] != "Changed name"
    assert len(upstream["requests"]) == before


@pytest.mark.parametrize(
    "version",
    [None, VERSION, "6701"],
    ids=["newest-local", "explicit-old", "explicit-new"],
)
def test_names_use_the_requested_local_version(upstream, lookup_source, version):
    lookup = lookup_source[0]
    lookup()
    changed_id = replace_name(upstream, lookup_source, "Changed name", "6701")
    data.update("6701")
    actual = lookup(version)[changed_id]
    assert (actual == "Changed name") == (version != VERSION)


def test_forced_reinstall_invalidates_cached_names(upstream, lookup_source):
    lookup = lookup_source[0]
    lookup()
    changed_id = replace_name(upstream, lookup_source, "Replacement name")
    data.update(VERSION, force=True)
    assert lookup()[changed_id] == "Replacement name"


def test_removal_invalidates_names_before_reacquisition(upstream, lookup_source):
    lookup = lookup_source[0]
    lookup()
    changed_id = replace_name(upstream, lookup_source, "Replacement name")
    data.remove(VERSION)
    before = len(upstream["requests"])
    assert lookup()[changed_id] == "Replacement name"
    assert len(upstream["requests"]) > before


@pytest.fixture
def corrupt_cached_names(upstream):
    hero_names()
    (data.BOON_DATA_DIR / VERSION / "heroes.json").write_text("corrupt")
    return upstream


def test_explicit_names_reject_a_corrupt_cached_catalog(corrupt_cached_names):
    with pytest.raises(data.DataError, match="boon get.*--force"):
        hero_names(VERSION)


def test_default_names_do_not_hide_corruption_when_offline(corrupt_cached_names):
    corrupt_cached_names["offline"] = True
    with pytest.raises(data.DataError, match="offline"):
        hero_names()


def test_unlocalized_hero_uses_its_internal_name(upstream):
    record = upstream["records"]["heroes"][0]
    record["display_name"] = None
    upstream["index"], upstream["files"] = release(records=upstream["records"])
    assert hero_names()[record["hero_id"]] == record["hero_name"]


@pytest.mark.parametrize("cached", [False, True], ids=["first-load", "cached"])
def test_catalog_names_reject_conflicting_ids(upstream, lookup_source, cached):
    lookup, catalog, _, label_key = lookup_source
    if cached:
        lookup()
    records = upstream["records"]
    records[catalog].append({**records[catalog][0], label_key: "Conflict"})
    upstream["index"], upstream["files"] = release(records=records)
    if cached:
        data.update(VERSION, force=True)
    with pytest.raises(data.DataError, match="conflicting"):
        lookup()


def test_removed_datasets_are_not_public():
    import boon

    assert not hasattr(boon.Demo, "healing")
    assert not hasattr(boon.Demo, "barriers")
    assert {"healing", "barriers"}.isdisjoint(boon.Demo.available_datasets())
    with pytest.raises(ModuleNotFoundError):
        __import__("boon.barriers")


def test_cli_players_reports_catalog_download_failure(upstream, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import polars as pl
    from boon import cli
    from typer.testing import CliRunner

    path = tmp_path / "demo.dem"
    path.touch()
    monkeypatch.setattr(
        cli,
        "_open",
        lambda path: SimpleNamespace(players=pl.DataFrame({"hero_id": [1]})),
    )

    def offline(url):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(data, "_request", offline)
    result = CliRunner().invoke(cli.app, ["players", str(path)])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "offline" in result.output


def test_breakable_retry_preserves_catalog_mapped_rows(upstream, monkeypatch):
    import polars as pl

    path = Path(__file__).parent / "fixtures" / "108575009.dem"
    if not path.exists():
        pytest.skip("108575009.dem fixture not available")
    # A changed label and an absent ID prove the dataset has no embedded fallback.
    records = upstream["records"]
    records["misc"] = [
        {
            "misc_id": 3986897915,
            "misc_name": "catalog_crate",
            "definition": {"_class": "citadel_breakable_prop"},
        }
    ]
    upstream["index"], upstream["files"] = release(records=records)
    demo = Demo(str(path), preload=False)

    def offline(url):
        raise urllib.error.URLError("offline")

    with monkeypatch.context() as context:
        context.setattr(data, "_request", offline)
        with pytest.raises(data.DataError, match="offline"):
            demo.load("breakables")

    # Retrying the same Demo preserves all events and their recorded IDs.
    frame = demo.breakables
    counts = {
        (row["subclass_id"], row["subclass_name"]): row["len"]
        for row in frame.group_by("subclass_id", "subclass_name").len().to_dicts()
    }
    assert counts == {
        (202631964, "BREAKABLE_NOT_FOUND"): 58,
        (3719077267, "BREAKABLE_NOT_FOUND"): 142,
        (3986897915, "catalog_crate"): 355,
    }
    assert frame.schema["subclass_id"] == pl.UInt32
