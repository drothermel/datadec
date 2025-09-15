import sys
from rich.console import Console

from datadec.fancy_table import FancyTable, HeaderCell


def test_multi_header_table():
    """Test the new multi-header functionality."""
    console = Console()

    # Create a FancyTable with proper multi-row headers
    table = FancyTable(title="Multi-Header Test Table")

    # Add columns
    table.add_column("Name")
    table.add_column("Age")
    table.add_column("Score1")
    table.add_column("Score2")

    # Add first header row with spanning
    table.add_header_row(
        table.create_spanned_header("Model Configuration", 2, "bold blue"),
        table.create_spanned_header("Learning Rates", 2, "bold green"),
    )

    # Add second header row with actual column names
    table.add_header_row("Model Size", "Dataset Tokens", "1e-04", "3e-04")

    # Add some data
    table.add_row("125M", "100M", "0.652", "0.698")
    table.add_row("350M", "100M", "0.678", "0.725")
    table.add_row("125M", "300M", "0.689", "0.734")

    console.print("\n=== Multi-Header Table Test ===")
    console.print(table)


def test_backward_compatibility():
    """Test that backward compatibility still works."""
    console = Console()

    table = FancyTable(title="Backward Compatibility Test")
    table.add_column("Column 1", style="cyan")
    table.add_column("Column 2", style="magenta")
    table.add_column("Column 3", style="green")

    table.add_row("Row 1", "Data A", "Value 1")
    table.add_row("Row 2", "Data B", "Value 2")

    console.print("\n=== Backward Compatibility Test ===")
    console.print(table)


if __name__ == "__main__":
    console = Console()
    console.print("[bold]Testing Multi-Header FancyTable Implementation[/bold]")

    try:
        test_multi_header_table()
        test_backward_compatibility()
        console.print("\n[bold green]✅ All tests completed successfully![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Test failed: {e}[/bold red]")
        import traceback

        console.print(traceback.format_exc())
