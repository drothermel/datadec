import pytest
from datadec.data import DataDecide

# Test configuration constants
TEST_DATA_DIR = "./data"


@pytest.fixture(scope="session")
def dd_instance():
    """Create a DataDecide instance for testing using the default data directory."""
    return DataDecide(data_dir=TEST_DATA_DIR)


@pytest.fixture
def sample_real_data(dd_instance):
    """Get a small subset of real DataDecide data for testing."""
    full_data = dd_instance.full_eval
    subset = full_data.head(200)
    return subset
