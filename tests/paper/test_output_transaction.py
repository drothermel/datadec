from pathlib import Path

import pytest

import datadec.paper.output_transaction as transaction
from datadec.paper.output_transaction import replace_output_set


def test_exact_directory_removes_stale_files(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    figures = tmp_path / "validation-figures"
    figures.mkdir()
    current = figures / "current.svg"
    stale = figures / "stale.svg"
    report.write_text("old report\n")
    current.write_text("old current\n")
    stale.write_text("stale\n")

    replace_output_set(
        ((report, "new report\n"), (current, b"<svg>new</svg>")),
        exact_directories={figures: (current,)},
    )

    assert report.read_text() == "new report\n"
    assert current.read_bytes() == b"<svg>new</svg>"
    assert tuple(figures.iterdir()) == (current,)


def test_exact_directory_failure_rolls_back_replacements_and_stale_removals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.md"
    figures = tmp_path / "validation-figures"
    figures.mkdir()
    current = figures / "current.svg"
    first_stale = figures / "a-stale.svg"
    second_stale = figures / "b-stale.svg"
    report.write_text("old report\n")
    current.write_text("old current\n")
    first_stale.write_text("first stale\n")
    second_stale.write_text("second stale\n")
    original_remove = transaction._remove_file

    def fail_second_stale(path: Path) -> None:
        if path == second_stale:
            raise OSError("injected stale removal failure")
        original_remove(path)

    monkeypatch.setattr(transaction, "_remove_file", fail_second_stale)

    with pytest.raises(OSError, match="injected stale removal failure"):
        replace_output_set(
            ((report, "new report\n"), (current, b"<svg>new</svg>")),
            exact_directories={figures: (current,)},
        )

    assert report.read_text() == "old report\n"
    assert current.read_text() == "old current\n"
    assert first_stale.read_text() == "first stale\n"
    assert second_stale.read_text() == "second stale\n"
    assert {path.name for path in figures.iterdir()} == {
        "a-stale.svg",
        "b-stale.svg",
        "current.svg",
    }
