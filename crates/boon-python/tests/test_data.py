"""JSON catalog acquisition, integrity checks, and offline CLI behavior."""

import copy
import json
from pathlib import Path

import pytest
from boon import data
from boon.cli import app
from catalog_helpers import (
    FILES,
    PUBLISHED,
    VERSION,
    VERSION_DATE,
    VERSION_TIME,
    catalog_upstream,
    fingerprint,
    release,
)
from typer.testing import CliRunner


@pytest.fixture
def upstream(monkeypatch, tmp_path):
    return catalog_upstream(monkeypatch, tmp_path / "cache")


def replace_manifest(upstream, manifest):
    content = json.dumps(manifest).encode()
    for record in (
        upstream["index"]["versions"][VERSION],
        upstream["index"]["snapshots"][VERSION],
    ):
        metadata = record["artifacts"]["manifest.json"]
        metadata.update(fingerprint(content))
        upstream["files"][metadata["url"]] = content


@pytest.fixture
def installed_version(upstream):
    return data.update(VERSION)


def test_download_installs_the_complete_json_release(upstream):
    directory = data.update()
    assert directory == data.BOON_DATA_DIR / VERSION
    assert data.available_files(VERSION) == sorted(FILES)
    assert {path.name for path in directory.iterdir()} == FILES | {".install.json"}
    for name in FILES:
        assert (directory / name).read_bytes() == upstream["files"][
            f"{data._DOWNLOAD_URL}/{VERSION}/{name}"
        ]


def test_manifest_retains_the_source_client_version(installed_version):
    assert data.manifest(VERSION)["client_version"] == VERSION


def test_explicit_get_reuses_the_installed_version_offline(upstream, installed_version):
    upstream["offline"] = True
    before = len(upstream["requests"])
    assert data.update(VERSION) == installed_version
    assert len(upstream["requests"]) == before


@pytest.mark.parametrize(
    ("name", "version", "filename"),
    [
        ("abilities", None, "abilities.json"),
        ("heroes.json", VERSION, "heroes.json"),
        ("misc", VERSION, "misc.json"),
    ],
)
def test_catalog_path_resolves_installed_json_offline(
    upstream, installed_version, name, version, filename
):
    upstream["offline"] = True
    before = len(upstream["requests"])
    assert data.catalog_path(name, version) == installed_version / filename
    assert len(upstream["requests"]) == before


def test_local_receipt_keeps_the_release_timestamp(installed_version):
    assert data.local_versions()[0]["released_at"] == PUBLISHED


def test_default_get_uses_latest_index_version_not_newest_local(upstream):
    data.update(VERSION)
    upstream["index"], upstream["files"] = release("6699")
    assert data.update().name == "6699"
    assert data.resolve_version() == "6699"
    assert [v["client_version"] for v in data.local_versions()] == ["6699", VERSION]


def test_shared_snapshot_is_installed_under_requested_client_version(upstream):
    entry = copy.deepcopy(upstream["index"]["versions"][VERSION])
    entry["client_version"] = "6699"
    upstream["index"]["versions"]["6699"] = entry
    upstream["index"]["latest"] = "6699"
    directory = data.update("6699")
    assert directory.name == "6699"
    assert data.manifest("6699")["client_version"] == VERSION
    assert data.local_versions()[0]["client_version"] == "6699"
    assert all("/6699/" not in url for url in upstream["requests"])


@pytest.mark.parametrize(
    "version",
    [
        "../escape",
        "/tmp/escape",
        "..",
        "6698-aaaaaaaaaaaa",
        "6698/child",
        "C:\\escape",
        "latest",
    ],
)
@pytest.mark.parametrize("operation", [data.update, data.remove])
def test_invalid_client_version_is_rejected_before_network(
    upstream, version, operation
):
    with pytest.raises(data.DataError, match="numeric client version"):
        operation(version)
    assert not upstream["requests"]


@pytest.mark.parametrize(
    "name",
    ["../heroes", "/tmp/heroes", "heroes/other", "heroes\\other", "heroes.vdata"],
)
def test_invalid_catalog_name_is_rejected_before_network(upstream, name):
    with pytest.raises(data.DataError, match="filename"):
        data.catalog_path(name, VERSION)
    assert not upstream["requests"]


def test_unknown_version_is_not_substituted(upstream):
    with pytest.raises(data.DataError, match="1234 is not available"):
        data.update("1234")
    assert upstream["requests"] == [data.INDEX_URL]
    assert not data.BOON_DATA_DIR.exists()


@pytest.mark.parametrize("damage", ["truncated", "checksum", "oversized", "offline"])
def test_failed_force_download_preserves_existing_installation(upstream, damage):
    directory = data.update(VERSION)
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    url = f"{data._DOWNLOAD_URL}/{VERSION}/heroes.json"
    content = upstream["files"][url]
    if damage == "truncated":
        upstream["files"][url] = content[:-1]
    elif damage == "checksum":
        upstream["files"][url] = b"x" * len(content)
    elif damage == "oversized":
        upstream["files"][url] = content + b"x"
    else:
        upstream["offline"] = True
    with pytest.raises(data.DataError):
        data.update(VERSION, force=True)
    assert original == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert sorted(p.name for p in data.BOON_DATA_DIR.iterdir()) == [VERSION]


@pytest.fixture
def corrupt_installation(installed_version):
    path = installed_version / "abilities.json"
    path.write_bytes(b"x" * path.stat().st_size)
    return installed_version


def test_corrupt_installation_is_not_listed_as_local(corrupt_installation):
    assert data.local_versions() == []


def test_reusing_corrupt_installation_requires_force(corrupt_installation):
    with pytest.raises(data.DataError, match="--force"):
        data.update(VERSION)


def test_versions_cli_marks_corrupt_installation_as_not_installed(corrupt_installation):
    result = CliRunner().invoke(app, ["versions", "--json"])
    assert json.loads(result.stdout)["versions"][0]["installed"] is False


def test_force_repairs_same_size_corruption(upstream, corrupt_installation):
    data.update(VERSION, force=True)
    assert len(data.local_versions()) == 1
    assert (corrupt_installation / "abilities.json").read_bytes() == upstream["files"][
        f"{data._DOWNLOAD_URL}/{VERSION}/abilities.json"
    ]


@pytest.mark.parametrize(
    "damage",
    ["missing", "source", "date", "size", "hash", "url", "filename", "snapshot"],
)
def test_invalid_index_is_rejected_before_asset_download(upstream, damage):
    index = upstream["index"]
    entry = index["versions"][VERSION]
    if damage == "missing":
        del entry["artifacts"]["heroes.json"]
    elif damage == "source":
        index["source_repository"] = "unexpected/repository"
    elif damage == "date":
        entry["released_at"] = "2026-09-22T00:31:49"
    elif damage == "size":
        entry["artifacts"]["heroes.json"]["bytes"] = -1
    elif damage == "hash":
        entry["artifacts"]["heroes.json"]["sha256"] = "no"
    elif damage == "url":
        entry["artifacts"]["heroes.json"]["url"] = "https://example.com/heroes.json"
    elif damage == "filename":
        entry["artifacts"]["../heroes.json"] = entry["artifacts"].pop("heroes.json")
    else:
        entry["snapshot"] = "../escape"
    with pytest.raises(data.DataError):
        data.update(VERSION)
    assert upstream["requests"] == [data.INDEX_URL]
    assert not data.BOON_DATA_DIR.exists()


@pytest.mark.parametrize(
    "damage", ["source", "snapshot", "origin", "heroes.json", "misc.json", "checksum"]
)
def test_manifest_is_checked_even_when_download_matches_index(upstream, damage):
    manifest = json.loads(
        upstream["files"][f"{data._DOWNLOAD_URL}/{VERSION}/manifest.json"]
    )
    if damage == "source":
        manifest["source"]["repository"] = "unexpected/repository"
    elif damage == "snapshot":
        manifest["release_key"] = "6699"
    elif damage == "origin":
        manifest["client_version"] = "6699"
    elif damage in {"heroes.json", "misc.json"}:
        del manifest["artifacts"][damage]
    else:
        manifest["artifacts"]["heroes.json"]["sha256"] = "0" * 64
    replace_manifest(upstream, manifest)
    match = "every catalog" if damage in {"heroes.json", "misc.json"} else None
    with pytest.raises(data.DataError, match=match):
        data.update(VERSION)
    assert data.local_versions() == []


def test_incomplete_cache_is_not_an_installation(upstream):
    directory = data.update(VERSION)
    (directory / "heroes.json").unlink()
    assert data.local_versions() == []


def test_symlink_installation_is_not_followed_or_replaced(upstream, tmp_path):
    target = tmp_path / "other"
    target.mkdir()
    data.BOON_DATA_DIR.mkdir()
    directory = data.BOON_DATA_DIR / VERSION
    directory.symlink_to(target, target_is_directory=True)
    for force in (False, True):
        with pytest.raises(data.DataError, match="symlink"):
            data.update(VERSION, force=force)
    assert directory.is_symlink()
    assert data.local_versions() == []
    assert list(target.iterdir()) == []


def test_force_install_rolls_back_on_rename_failure(upstream, monkeypatch):
    directory = data.update(VERSION)
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    rename = Path.rename

    def fail_stage(path, target):
        if path.name == "snapshot":
            raise OSError("rename failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_stage)
    with pytest.raises(data.DataError, match="rename failed"):
        data.update(VERSION, force=True)
    assert original == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert len(data.local_versions()) == 1


def test_versions_json_reports_build_time_before_installation(upstream):
    result = CliRunner().invoke(app, ["versions", "--json"])
    assert json.loads(result.stdout)["versions"][0] == {
        "client_version": VERSION,
        "released_at": PUBLISHED,
        "version_date": VERSION_DATE,
        "version_time": VERSION_TIME,
        "installed": False,
        "path": None,
    }


def test_get_json_reports_downloaded_artifacts(upstream):
    result = CliRunner().invoke(app, ["get", VERSION, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["client_version"] == VERSION
    assert set(payload["files"]) == FILES


def test_versions_json_reports_the_installed_path(installed_version):
    result = CliRunner().invoke(app, ["versions", "--json"])
    assert result.exit_code == 0, result.output
    row = json.loads(result.stdout)["versions"][0]
    assert row["installed"] is True
    assert row["path"] == str(installed_version)


def test_versions_text_shows_build_time_and_installation(installed_version):
    result = CliRunner().invoke(app, ["versions"])
    assert all(
        text in result.stdout for text in [VERSION, VERSION_DATE, VERSION_TIME, "yes"]
    )
    assert "Released at" not in result.stdout and PUBLISHED not in result.stdout


def test_versions_falls_back_to_local_receipt_when_offline(upstream, installed_version):
    upstream["offline"] = True
    result = CliRunner().invoke(app, ["versions"])
    assert result.exit_code == 0, result.output
    assert all(text in result.stdout for text in [VERSION, VERSION_DATE, VERSION_TIME])
    assert "Released at" not in result.stdout and PUBLISHED not in result.stdout
    assert "could not list" in result.output


@pytest.mark.parametrize(
    "args",
    [["versions", "--local"], ["get", VERSION]],
    ids=["local-versions", "explicit-get"],
)
def test_explicit_local_commands_do_not_use_network(upstream, installed_version, args):
    upstream["offline"] = True
    before = len(upstream["requests"])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(upstream["requests"]) == before
    if args[0] == "versions":
        assert VERSION_DATE in result.stdout and VERSION_TIME in result.stdout
        assert "Released at" not in result.stdout and PUBLISHED not in result.stdout
    else:
        assert "5 JSON files" in result.stdout


def test_default_get_fails_when_the_remote_index_is_offline(
    upstream, installed_version
):
    upstream["offline"] = True
    result = CliRunner().invoke(app, ["get"])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_empty_local_cache_never_uses_network(upstream):
    result = CliRunner().invoke(app, ["versions", "--local"])
    assert result.exit_code == 0
    assert "no installed versions" in result.stdout
    assert not upstream["requests"]


def test_remote_failure_without_installed_versions_returns_error(upstream):
    upstream["offline"] = True
    result = CliRunner().invoke(app, ["versions", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["versions"] == []
    assert "offline" in json.loads(result.stdout)["error"]


def test_permission_error_listing_cache_is_reported(upstream, monkeypatch):
    data.BOON_DATA_DIR.mkdir()

    def denied(path):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "iterdir", denied)
    result = CliRunner().invoke(app, ["versions", "--local"])
    assert result.exit_code == 1
    assert "could not list the boon-data cache" in result.output


@pytest.fixture
def removed_version(upstream):
    directory = data.update(VERSION)
    upstream["index"], upstream["files"] = release("6699")
    other = data.update("6699")
    original = {p.name: p.read_bytes() for p in other.iterdir()}
    upstream["offline"] = True
    before = len(upstream["requests"])
    result = CliRunner().invoke(app, ["remove", VERSION, "--json"])
    assert result.exit_code == 0, result.output
    return directory, other, original, result, before


def test_remove_cli_reports_the_removed_version(removed_version):
    directory, _, _, result, _ = removed_version
    assert json.loads(result.stdout) == {
        "client_version": VERSION,
        "path": str(directory),
        "removed": True,
    }


def test_remove_deletes_only_the_selected_directory(removed_version):
    directory, _, _, _, _ = removed_version
    assert not directory.exists()
    assert data.BOON_DATA_DIR.is_dir()


def test_remove_preserves_other_installed_versions(removed_version):
    _, other, original, _, _ = removed_version
    assert {p.name: p.read_bytes() for p in other.iterdir()} == original
    assert [entry["client_version"] for entry in data.local_versions()] == ["6699"]


def test_remove_needs_no_network_requests(removed_version, upstream):
    _, _, _, _, before = removed_version
    assert len(upstream["requests"]) == before


@pytest.fixture
def removed_corrupt_version(upstream):
    directory = data.update(VERSION)
    (directory / "heroes.json").unlink()
    (directory / ".install.json").write_text("invalid")
    upstream["offline"] = True
    removed = CliRunner().invoke(app, ["remove", VERSION])
    return directory, removed


def test_remove_accepts_a_corrupt_installation(removed_corrupt_version):
    directory, removed = removed_corrupt_version
    assert removed.exit_code == 0 and f"Removed {VERSION}" in removed.stdout
    assert not directory.exists()


def test_get_reinstalls_a_removed_corrupt_version(removed_corrupt_version, upstream):
    directory, removed = removed_corrupt_version
    assert removed.exit_code == 0, removed.output
    upstream["offline"] = False
    downloaded = CliRunner().invoke(app, ["get", VERSION])
    assert downloaded.exit_code == 0, downloaded.output
    assert "5 JSON files" in downloaded.stdout
    assert (directory / "misc.json").is_file()
    assert len(data.local_versions()) == 1


@pytest.mark.parametrize("existing_file", [False, True])
def test_remove_missing_directory_or_regular_file_is_an_error(upstream, existing_file):
    path = data.BOON_DATA_DIR / VERSION
    if existing_file:
        path.parent.mkdir()
        path.write_text("not a catalog directory")
    result = CliRunner().invoke(app, ["remove", VERSION])
    assert result.exit_code == 1 and "not installed" in result.output
    assert path.exists() == existing_file
    assert not upstream["requests"]


def test_remove_requires_an_explicit_version(upstream):
    result = CliRunner().invoke(app, ["remove"])
    assert result.exit_code == 2
    assert not upstream["requests"]


@pytest.fixture
def external_marker(tmp_path):
    target = tmp_path / "external"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep")
    data.BOON_DATA_DIR.mkdir()
    return target, marker


def test_remove_rejects_a_symlinked_version(upstream, external_marker):
    target, marker = external_marker
    path = data.BOON_DATA_DIR / VERSION
    path.symlink_to(target, target_is_directory=True)
    with pytest.raises(data.DataError, match="symlink"):
        data.remove(VERSION)
    assert path.is_symlink() and marker.read_text() == "keep"
    assert not upstream["requests"]


def test_remove_does_not_follow_nested_symlinks(upstream, external_marker):
    target, marker = external_marker
    path = data.BOON_DATA_DIR / VERSION
    path.mkdir()
    (path / "nested-link").symlink_to(target, target_is_directory=True)
    data.remove(VERSION)
    assert not path.exists() and marker.read_text() == "keep"
    assert not upstream["requests"]


def test_remove_reports_filesystem_errors(upstream, monkeypatch):
    directory = data.update(VERSION)

    def denied(path):
        raise PermissionError("denied")

    monkeypatch.setattr(data.shutil, "rmtree", denied)
    result = CliRunner().invoke(app, ["remove", VERSION])
    assert result.exit_code == 1 and "could not remove" in result.output
    assert directory.is_dir()


def test_backfill_only_index_uses_highest_client_version(upstream):
    upstream["index"]["latest"] = None
    older, _ = release("999")
    upstream["index"]["versions"].update(older["versions"])
    upstream["index"]["snapshots"].update(older["snapshots"])
    assert data.latest_version() == VERSION
    assert data.update().name == VERSION
