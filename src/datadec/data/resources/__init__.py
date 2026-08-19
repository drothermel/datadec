from importlib.resources import files
from importlib.resources.abc import Traversable

DATASET_FEATURES_CSV: Traversable = files(__name__).joinpath("dataset_features.csv")

__all__ = ["DATASET_FEATURES_CSV"]
