"""Teste la sauvegarde des logs dans la base de données."""

from datetime import UTC, datetime

import pytest

from src.database import database
from src.tests.conftest import EVENT_TIME, VALID_PROFILE


def test_save_prediction_log():
    database.model_logs.reset_mock()

    request_id = database.save_prediction_log(
        requested_params=VALID_PROFILE,
        predicted_class=0,
        execution_time_ms=12.5,
        event_time=datetime.fromisoformat(EVENT_TIME).replace(tzinfo=UTC),
    )

    assert request_id == 42
    database.model_logs.insert.assert_called_once()


def test_save_error_log():
    database.model_logs.reset_mock()

    request_id = database.save_error_log(
        requested_params={"AMT_CREDIT": "bad"},
        error_message="Validation error",
        execution_time_ms=3.0,
        event_time=datetime.fromisoformat(EVENT_TIME).replace(tzinfo=UTC),
    )

    assert request_id == 42
    assert database.model_logs.insert.call_count >= 1
