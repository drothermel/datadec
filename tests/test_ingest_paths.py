from __future__ import annotations

from pathlib import Path

from datadec.ingest.paths import DATADEC_DATA_DIR_ENV, Paths


def test_paths_default_uses_home(monkeypatch) -> None:
    monkeypatch.delenv(DATADEC_DATA_DIR_ENV, raising=False)
    paths = Paths()
    expected = Path.home() / "data"
    assert paths.data_dir == expected
    assert paths.data_cache_dir == expected / "cache"
    assert paths.metrics_all_dir == expected / "datadec"


def test_paths_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATADEC_DATA_DIR_ENV, str(tmp_path))
    paths = Paths()
    assert paths.data_dir == tmp_path
    assert paths.data_cache_dir == tmp_path / "cache"
    assert paths.metrics_all_dir == tmp_path / "datadec"
