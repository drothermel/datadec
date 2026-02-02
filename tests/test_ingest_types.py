from __future__ import annotations


from datadec.ingest.types import TaskArtifactType


class TestTaskArtifactType:
    def test_predictions_extension(self) -> None:
        assert TaskArtifactType.PREDICTIONS.extension == "jsonl"

    def test_recorded_inputs_extension(self) -> None:
        assert TaskArtifactType.RECORDED_INPUTS.extension == "jsonl"

    def test_requests_extension(self) -> None:
        assert TaskArtifactType.REQUESTS.extension == "jsonl"

    def test_config_extension(self) -> None:
        assert TaskArtifactType.CONFIG.extension == "json"

    def test_metrics_extension(self) -> None:
        assert TaskArtifactType.METRICS.extension == "json"

    def test_predictions_filename_pattern(self) -> None:
        assert TaskArtifactType.PREDICTIONS.filename_pattern == "*-predictions.jsonl"

    def test_recorded_inputs_filename_pattern(self) -> None:
        assert TaskArtifactType.RECORDED_INPUTS.filename_pattern == "*-recorded-inputs.jsonl"

    def test_requests_filename_pattern(self) -> None:
        assert TaskArtifactType.REQUESTS.filename_pattern == "*-requests.jsonl"

    def test_config_filename_pattern(self) -> None:
        assert TaskArtifactType.CONFIG.filename_pattern == "*-config.json"

    def test_metrics_filename_pattern(self) -> None:
        assert TaskArtifactType.METRICS.filename_pattern == "*-metrics.json"

    def test_enum_values(self) -> None:
        assert TaskArtifactType.PREDICTIONS.value == "predictions"
        assert TaskArtifactType.RECORDED_INPUTS.value == "recorded-inputs"
        assert TaskArtifactType.REQUESTS.value == "requests"
        assert TaskArtifactType.CONFIG.value == "config"
        assert TaskArtifactType.METRICS.value == "metrics"
