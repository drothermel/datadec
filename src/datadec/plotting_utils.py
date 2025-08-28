"""Utilities for dr_plotter integration (optional dependency)."""

import sys
from typing import Any, Tuple


def check_plotting_available() -> bool:
    """Check if dr_plotter is available and provide helpful error."""
    try:
        import dr_plotter  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "Plotting functionality requires the 'dr_plotter' optional dependency.\n"
            "Install with: uv add 'datadec[plotting]' or pip install 'datadec[plotting]'"
        ) from None


def safe_import_plotting() -> Tuple[Any, Any, Any]:
    """Import dr_plotter with helpful error if not available."""
    check_plotting_available()
    from dr_plotter import FigureManager
    from dr_plotter.figure_config import FigureConfig
    from dr_plotter.legend_manager import LegendConfig, LegendStrategy
    return FigureManager, FigureConfig, (LegendConfig, LegendStrategy)


def require_plotting():
    """Check plotting availability and exit with helpful message if missing."""
    try:
        check_plotting_available()
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)