#!/usr/bin/env python3
"""Test script for cell spanning functionality."""

import sys

from rich.console import Console

# Add the src directory to Python path
sys.path.insert(0, "src")

from datadec.fancy_table import FancyTable


def test_more_basic_spanning():
    """Test basic cell spanning functionality."""
    console = Console()

    console.print("\n=== Basic Cell Spanning Test ===")

    # Create a FancyTable with spanning headers
    table = FancyTable(title="Cell Spanning Test")
    table.add_column("Col1")
    table.add_column("Col2")

    # Add header row with spans
    table.add_header_row(
        table.create_spanned_header("Group A", 2),  # Spans columns 0,1
    )

    # Add second header row with individual column names
    table.add_header_row("Name", "Age")

    # Add data rows
    table.add_row("Alice", "25")
    table.add_row("Bob", "30")
    table.add_row("Charlie", "28")

    console.print(table)


def test_basic_spanning():
    """Test basic cell spanning functionality."""
    console = Console()

    console.print("\n=== Basic Cell Spanning Test ===")

    # Create a FancyTable with spanning headers
    table = FancyTable(title="Cell Spanning Test")
    table.add_column("Col1")
    table.add_column("Col2")
    table.add_column("Col3")
    table.add_column("Col4")

    # Add header row with spans
    table.add_header_row(
        table.create_spanned_header("Group A", 2),  # Spans columns 0,1
        table.create_spanned_header("Group B", 2),  # Spans columns 2,3
    )

    # Add second header row with individual column names
    table.add_header_row("Name", "Age", "Score", "Rank")

    # Add data rows
    table.add_row("Alice", "25", "95", "1")
    table.add_row("Bob", "30", "87", "2")
    table.add_row("Charlie", "28", "92", "3")

    console.print(table)


def test_mixed_spanning():
    """Test mixed spanning (some cells span, others don't)."""
    console = Console()

    console.print("\n=== Mixed Spanning Test ===")

    table = FancyTable(title="Mixed Spanning")
    table.add_column("A")
    table.add_column("B")
    table.add_column("C")
    table.add_column("D")
    table.add_column("E")

    # First row: span + individual + span
    table.add_header_row(
        table.create_spanned_header("First Group", 2),  # Spans A,B
        "Single",  # Just C
        table.create_spanned_header("Second Group", 2),  # Spans D,E
    )

    # Second row: all individual
    table.add_header_row("A1", "B1", "C1", "D1", "E1")

    table.add_row("a", "b", "c", "d", "e")
    table.add_row("1", "2", "3", "4", "5")

    console.print(table)


def test_different_span_lengths():
    """Test different span lengths."""
    console = Console()

    console.print("\n=== Different Span Lengths Test ===")

    table = FancyTable(title="Variable Spans")
    for i in range(6):
        table.add_column(f"Col{i + 1}")

    # Test different span lengths
    table.add_header_row(
        table.create_spanned_header("Span 3", 3),  # Spans 3 columns
        table.create_spanned_header("Span 2", 2),  # Spans 2 columns
        "Single",  # 1 column
    )

    table.add_header_row("A", "B", "C", "D", "E", "F")

    table.add_row("1", "2", "3", "4", "5", "6")
    table.add_row("x", "y", "z", "p", "q", "r")

    console.print(table)


if __name__ == "__main__":
    console = Console()
    console.print("[bold]Testing Cell Spanning Implementation[/bold]")

    try:
        test_more_basic_spanning()
        test_basic_spanning()
        test_mixed_spanning()
        test_different_span_lengths()
        console.print("\n[bold green]✅ All spanning tests completed![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Test failed: {e}[/bold red]")
        import traceback

        console.print(traceback.format_exc())
