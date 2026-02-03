import warnings

from datadec.data import DataDecide
from datadec import ingest
from datadec import analysis

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

__all__ = [
    "DataDecide",
    "ingest",
    "analysis",
]
