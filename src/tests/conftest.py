"""Fixtures partagées et mocks SQLAlchemy avant import des modules métier."""

from __future__ import annotations

import pickle as _pickle_module
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Mock SQLAlchemy avant tout import de src.database.database
_mock_connection = MagicMock()
_mock_connection.__enter__ = MagicMock(return_value=_mock_connection)
_mock_connection.__exit__ = MagicMock(return_value=False)

_mock_execute_result = MagicMock()
_mock_execute_result.inserted_primary_key = (42,)
_mock_connection.execute.return_value = _mock_execute_result

_mock_engine = MagicMock()
_mock_engine.connect.return_value = _mock_connection
_mock_engine.begin.return_value = _mock_connection

_mock_insert_stmt = MagicMock()
_mock_insert_stmt.values.return_value = MagicMock()
_mock_table = MagicMock()
_mock_table.insert.return_value = _mock_insert_stmt

patch("sqlalchemy.create_engine", return_value=_mock_engine).start()
patch("sqlalchemy.Table", return_value=_mock_table).start()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "src" / "model" / "lgb_model.pkl"
ONNX_MODEL_PATH = BASE_DIR / "src" / "model" / "onnx_model.onnx"
DATA_PATH = BASE_DIR / "data" / "original" / "demonstration_data.csv"

VALID_PROFILE = {
    "EXT_SOURCE_1": 0.7526145,
    "EXT_SOURCE_2": 0.7896544,
    "EXT_SOURCE_3": 0.15951954,
    "client_installments_AMT_PAYMENT_min_sum": 27746.775,
    "DAYS_EMPLOYED": -2329.0,
    "AMT_CREDIT": 568800.0,
    "DAYS_BIRTH": -19241,
    "previous_CNT_PAYMENT_mean": 8.0,
    "bureau_DAYS_CREDIT_max": -49.0,
    "client_cash_CNT_INSTALMENT_FUTURE_min_mean": 0.75,
}

EVENT_TIME = "2026-03-15T10:30:00"


def _build_test_model() -> Pipeline:
    """Pipeline minimal compatible avec demonstration_data.csv pour les tests."""
    data = pd.read_csv(DATA_PATH, sep=";")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("classifier", DummyClassifier(strategy="constant", constant=0)),
        ]
    )
    pipeline.fit(data.head(10), [0] * 10)
    return pipeline


_original_pickle_load = _pickle_module.load


def _patched_pickle_load(file):
    file_name = getattr(file, "name", "")
    if "lgb_model.pkl" in str(file_name).replace("\\", "/"):
        return _build_test_model()
    return _original_pickle_load(file)


_pickle_module.load = _patched_pickle_load


@pytest.fixture
def sample_logs_df() -> pd.DataFrame:
    """DataFrame de logs après traitement load_logs (sans colonne day)."""
    return pd.DataFrame(
        {
            "request_id": [1, 2, 3, 4, 5],
            "ts": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-02 09:00:00",
                    "2026-01-02 10:00:00",
                    "2026-01-03 08:00:00",
                ],
                utc=True,
            ),
            "error": [False, True, False, False, True],
            "error_message": [None, "Validation error", None, None, "No features value was recieved"],
            "pred_class": [0, None, 1, 0, None],
            "execution_time_ms": [10.0, 5.0, 15.0, 20.0, None],
        }
    )


@pytest.fixture
def logs_df_with_day(sample_logs_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_logs_df.copy()
    df["day"] = df["ts"].dt.floor("D")
    return df


@pytest.fixture
def mock_save_prediction_log(monkeypatch):
    calls: list[dict] = []

    def _fake_save(**kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr(
        "src.api.scoring_api.save_prediction_log",
        _fake_save,
    )
    return calls


@pytest.fixture
def mock_save_error_log(monkeypatch):
    calls: list[dict] = []

    def _fake_save(**kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr(
        "src.api.scoring_api.save_error_log",
        _fake_save,
    )
    return calls


@pytest.fixture
def scoring_model():
    return _build_test_model()


@pytest.fixture
def explicative_features():
    schema_path = BASE_DIR / "data" / "schema" / "typeAdapters.json"
    return pd.read_json(schema_path, typ="series")
