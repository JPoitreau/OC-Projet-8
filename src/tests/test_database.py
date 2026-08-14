"""Teste la sauvegarde des logs dans la base de données."""

import json
from datetime import UTC, datetime

import pandas as pd

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


def test_save_prediction_log_remote_csv(tmp_path, monkeypatch):
    """En mode distant, les logs doivent être ajoutés au csv de remplacement."""
    csv_path = tmp_path / "model_logs_remote.csv"
    monkeypatch.setattr(database, "IS_REMOTE", True)
    monkeypatch.setattr(database, "REMOTE_LOGS_PATH", csv_path)

    request_id = database.save_prediction_log(
        requested_params=VALID_PROFILE,
        predicted_class=0,
        execution_time_ms=12.5,
        event_time=datetime.fromisoformat(EVENT_TIME).replace(tzinfo=UTC),
    )

    assert request_id == 1
    assert csv_path.is_file()

    logs = pd.read_csv(csv_path)
    assert list(logs.columns) == database.LOG_COLUMNS
    assert logs.loc[0, "request_id"] == 1
    assert logs.loc[0, "pred_class"] == 0
    assert not logs.loc[0, "error"]
    assert json.loads(logs.loc[0, "requested_params"]) == VALID_PROFILE


def test_save_error_log_remote_csv_increments_request_id(tmp_path, monkeypatch):
    """Chaque nouvel appel doit incrémenter request_id à partir du csv existant."""
    csv_path = tmp_path / "model_logs_remote.csv"
    monkeypatch.setattr(database, "IS_REMOTE", True)
    monkeypatch.setattr(database, "REMOTE_LOGS_PATH", csv_path)

    first_id = database.save_prediction_log(
        requested_params=VALID_PROFILE,
        predicted_class=1,
        execution_time_ms=5.0,
        event_time=datetime.fromisoformat(EVENT_TIME).replace(tzinfo=UTC),
    )
    second_id = database.save_error_log(
        requested_params={"AMT_CREDIT": "bad"},
        error_message="Validation error",
        execution_time_ms=3.0,
        event_time=datetime.fromisoformat(EVENT_TIME).replace(tzinfo=UTC),
    )

    assert (first_id, second_id) == (1, 2)

    logs = pd.read_csv(csv_path)
    assert len(logs) == 2
    assert logs.loc[1, "error"]
    assert logs.loc[1, "error_message"] == "Validation error"
