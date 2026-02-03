from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def _noop_add_marimo_display() -> Callable[[T], T]:
    def decorator(obj: T) -> T:
        return obj

    return decorator


try:
    from marimo_utils.display import add_marimo_display as _add_marimo_display
except Exception:  # pragma: no cover - fallback when marimo_utils isn't installed
    _add_marimo_display = _noop_add_marimo_display

add_marimo_display = _add_marimo_display

__all__ = ["add_marimo_display"]
