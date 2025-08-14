"""DataFrame loading and caching utilities for DataDecide.

This module provides a generic DataFrameLoader class that handles lazy loading
and caching of parquet files, eliminating repetitive property code.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class DataFrameLoader:
    """Handles lazy loading and caching of DataFrames from parquet files.

    This class provides a centralized mechanism for loading DataFrames on-demand
    and caching them in memory to avoid repeated disk reads.
    """

    def __init__(self):
        """Initialize an empty cache for storing loaded DataFrames."""
        self._cache: Dict[str, pd.DataFrame] = {}

    def load(self, path: Path, cache_key: Optional[str] = None) -> pd.DataFrame:
        """Load a DataFrame from a parquet file with caching.

        Args:
            path: Path to the parquet file to load
            cache_key: Optional custom key for caching. If None, uses str(path)

        Returns:
            Loaded pandas DataFrame

        Raises:
            FileNotFoundError: If the parquet file doesn't exist
        """
        key = cache_key or str(path)

        if key not in self._cache:
            if not path.exists():
                raise FileNotFoundError(f"Parquet file not found: {path}")
            self._cache[key] = pd.read_parquet(path)

        return self._cache[key]

    def is_cached(self, cache_key: str) -> bool:
        """Check if a DataFrame is already cached.

        Args:
            cache_key: Key to check in cache

        Returns:
            True if the key exists in cache, False otherwise
        """
        return cache_key in self._cache

    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """Clear cached DataFrames.

        Args:
            cache_key: Specific key to remove. If None, clears entire cache
        """
        if cache_key is None:
            self._cache.clear()
        else:
            self._cache.pop(cache_key, None)

    def cache_dataframe(self, df: pd.DataFrame, cache_key: str) -> None:
        """Manually cache a DataFrame.

        Args:
            df: DataFrame to cache
            cache_key: Key to use for caching
        """
        self._cache[cache_key] = df

    def get_cache_size(self) -> int:
        """Get the number of cached DataFrames.

        Returns:
            Number of DataFrames currently in cache
        """
        return len(self._cache)
