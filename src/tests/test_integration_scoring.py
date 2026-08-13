"""Teste l'enchaînement complets API → logs → base de données."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.tests.conftest import EVENT_TIME, VALID_PROFILE


@pytest.fixture
def scoring_api():
    import src.api.scoring_api as module

    return module


def test_integration_success_path(
    scoring_api,
    mock_save_prediction_log,
):
    result = scoring_api.process_scoring_request(
        VALID_PROFILE,
        scoring_api.scoring_model,
        simulated_event_time=EVENT_TIME,
        persist=True,
    )

    assert result["prediction"][0] in (0, 1)
    assert result["database_status"].startswith("✅")
    assert len(mock_save_prediction_log) == 1
    assert mock_save_prediction_log[0]["requested_params"] == pytest.approx(
        VALID_PROFILE,
        rel=0,
        abs=0,
    )


def test_integration_validation_to_error_log(
    scoring_api,
    mock_save_error_log,
):
    with pytest.raises(ValueError):
        scoring_api.process_scoring_request(
            {"DAYS_BIRTH": "not-int"},
            scoring_api.scoring_model,
            simulated_event_time=EVENT_TIME,
        )

    assert "Validation error" in mock_save_error_log[0]["error_message"]


def test_integration_empty_request_logs_error(
    scoring_api,
    mock_save_error_log,
):
    with pytest.raises(ValueError, match="No features value"):
        scoring_api.process_scoring_request(
            {},
            scoring_api.scoring_model,
            simulated_event_time=EVENT_TIME,
        )

    assert mock_save_error_log[0]["error_message"].startswith("No features value")


def test_integration_database_save_prediction():
    from src.database import database

    database.model_logs.reset_mock()
    database.model_logs.insert.return_value.values.return_value = MagicMock()

    request_id = database.save_prediction_log(
        requested_params=VALID_PROFILE,
        predicted_class=1,
        execution_time_ms=8.0,
        event_time=datetime.fromisoformat(EVENT_TIME).replace(tzinfo=UTC),
    )

    assert request_id == 42
    database.engine.begin.assert_called()


def test_integration_database_save_error():
    from src.database import database

    database.model_logs.reset_mock()

    request_id = database.save_error_log(
        requested_params={},
        error_message="Inference failed",
        execution_time_ms=0.0,
    )

    assert request_id == 42
    database.engine.begin.assert_called()
