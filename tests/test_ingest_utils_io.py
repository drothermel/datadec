from __future__ import annotations

from pathlib import Path

import pytest

from datadec.ingest.utils.io import iter_file_glob_from_roots


class TestIterFileGlobFromRoots:
    def test_finds_files_with_glob(self, tmp_path: Path) -> None:
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "file3.json").touch()

        result = list(iter_file_glob_from_roots([tmp_path], "*.txt"))
        assert len(result) == 2
        assert all(p.suffix == ".txt" for p in result)

    def test_finds_files_recursively(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").touch()
        (subdir / "nested.txt").touch()

        result = list(iter_file_glob_from_roots([tmp_path], "*.txt"))
        assert len(result) == 2

    def test_multiple_roots(self, tmp_path: Path) -> None:
        root1 = tmp_path / "root1"
        root2 = tmp_path / "root2"
        root1.mkdir()
        root2.mkdir()
        (root1 / "file1.txt").touch()
        (root2 / "file2.txt").touch()

        result = list(iter_file_glob_from_roots([root1, root2], "*.txt"))
        assert len(result) == 2

    def test_single_path_string(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").touch()

        result = list(iter_file_glob_from_roots(str(tmp_path), "*.txt"))
        assert len(result) == 1

    def test_single_path_object(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").touch()

        result = list(iter_file_glob_from_roots(tmp_path, "*.txt"))
        assert len(result) == 1

    def test_raises_for_missing_root(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            list(iter_file_glob_from_roots([missing], "*.txt"))

    def test_raises_for_file_as_root(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        file_path.touch()
        with pytest.raises(NotADirectoryError):
            list(iter_file_glob_from_roots([file_path], "*.txt"))

    def test_empty_result_for_no_matches(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").touch()
        result = list(iter_file_glob_from_roots([tmp_path], "*.json"))
        assert result == []
