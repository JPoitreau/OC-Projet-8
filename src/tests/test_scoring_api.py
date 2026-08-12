"""Teste la validation des paramètres, l'inférence et le traitement de la requête."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.tests.conftest import EVENT_TIME, ONNX_MODEL_PATH, VALID_PROFILE


@pytest.fixture
def scoring_api():
    import src.api.scoring_api as module

    return module


def test_update_table_none(scoring_api):
    result = scoring_api.update_table(None)

    assert list(result.columns) == ["feature", "value"]
    assert result.empty


def test_update_table_single_str(scoring_api):
    result = scoring_api.update_table("AMT_CREDIT")

    assert len(result) == 1
    assert result.iloc[0]["feature"] == "AMT_CREDIT"


def test_update_table_list(scoring_api):
    result = scoring_api.update_table(["AMT_CREDIT", "DAYS_BIRTH"])

    assert len(result) == 2


def test_validate_params_empty(scoring_api):
    message, parsed = scoring_api.validate_params(
        pd.DataFrame(columns=["feature", "value"]),
        scoring_api.explicative_features,
    )

    assert parsed is None
    assert "No feature selected" in message


def test_validate_params_unknown_feature(scoring_api):
    df = pd.DataFrame({"feature": ["UNKNOWN"], "value": [1]})

    message, parsed = scoring_api.validate_params(
        df,
        scoring_api.explicative_features,
    )

    assert parsed is None
    assert "Unknown feature" in message


def test_validate_params_invalid_value(scoring_api):
    df = pd.DataFrame({"feature": ["DAYS_BIRTH"], "value": ["not-an-int"]})

    message, parsed = scoring_api.validate_params(
        df,
        scoring_api.explicative_features,
    )

    assert parsed is None
    assert "Validation error" in message


def test_validate_params_success(scoring_api):
    df = pd.DataFrame(
        {
            "feature": ["AMT_CREDIT", "DAYS_BIRTH"],
            "value": [6000.0, -140],
        }
    )

    message, parsed = scoring_api.validate_params(
        df,
        scoring_api.explicative_features,
    )

    assert parsed is not None
    assert "validated" in message.lower()
    assert parsed["AMT_CREDIT"] == 6000.0
    assert parsed["DAYS_BIRTH"] == -140


def test_infer_from_new_vector_without_persist(
    scoring_api,
    scoring_model,
):
    prediction, message = scoring_api.infer_from_new_vector(
        VALID_PROFILE,
        scoring_model,
        persist=False,
    )

    assert prediction is not None
    assert prediction[0] in (0, 1)
    assert "not registered" in message


def test_infer_from_new_vector_with_persist(
    scoring_api,
    scoring_model,
    mock_save_prediction_log,
):
    prediction, message = scoring_api.infer_from_new_vector(
        VALID_PROFILE,
        scoring_model,
        persist=True,
    )

    assert prediction is not None
    assert len(mock_save_prediction_log) == 1
    assert "registered" in message


def test_infer_from_new_vector_onnx(
    scoring_api,
    mock_save_prediction_log,
):
    prediction, message = scoring_api.infer_from_new_vector(
        VALID_PROFILE,
        ONNX_MODEL_PATH,
        persist=False,
        onnx=True,
    )

    assert prediction is not None
    assert prediction[0] in (0, 1)
    assert len(mock_save_prediction_log) == 0


def test_infer_from_new_vector_error(
    scoring_api,
    mock_save_error_log,
):
    bad_model = MagicMock()
    bad_model.predict.side_effect = ValueError("predict failed")

    prediction, message = scoring_api.infer_from_new_vector(
        VALID_PROFILE,
        bad_model,
        persist=True,
    )

    assert prediction is None
    assert "predict failed" in message
    assert len(mock_save_error_log) == 1


def test_process_scoring_request_success(
    scoring_api,
    scoring_model,
    mock_save_prediction_log,
):
    result = scoring_api.process_scoring_request(
        VALID_PROFILE,
        scoring_model,
        simulated_event_time=EVENT_TIME,
        persist=True,
    )

    assert result["prediction"] is not None
    assert set(result["validated_params"].keys()) == set(VALID_PROFILE.keys())
    assert len(mock_save_prediction_log) == 1


def test_process_scoring_request_empty_input(
    scoring_api,
    scoring_model,
    mock_save_error_log,
):
    with pytest.raises(ValueError, match="No features value"):
        scoring_api.process_scoring_request(
            {},
            scoring_model,
            simulated_event_time=EVENT_TIME,
        )

    assert len(mock_save_error_log) == 1


def test_process_scoring_request_validation_error(
    scoring_api,
    scoring_model,
    mock_save_error_log,
):
    invalid_profile = {"AMT_CREDIT": "invalid", "DAYS_BIRTH": -140}

    with pytest.raises(ValueError, match="Validation error"):
        scoring_api.process_scoring_request(
            invalid_profile,
            scoring_model,
            simulated_event_time=EVENT_TIME,
        )

    assert len(mock_save_error_log) == 1
