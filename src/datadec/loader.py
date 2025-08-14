from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class DataFrameLoader:
    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def load(self, path: Path, cache_key: Optional[str] = None) -> pd.DataFrame:
        key = cache_key or str(path)

        if key not in self._cache:
            self._cache[key] = pd.read_parquet(path)

        return self._cache[key]

    def is_cached(self, cache_key: str) -> bool:
        return cache_key in self._cache

    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        if cache_key is None:
            self._cache.clear()
        else:
            self._cache.pop(cache_key, None)

    def cache_dataframe(self, df: pd.DataFrame, cache_key: str) -> None:
        self._cache[cache_key] = df

    def get_cache_size(self) -> int:
        return len(self._cache)
