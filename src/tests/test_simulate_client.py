"""Teste la simulation de profils clients et la corruption aléatoire des données."""

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import pytest

from src.tests.conftest import BASE_DIR, DATA_PATH
from src.database.simulate_client import maybe_corrupted_profile, random_datetime_between


def test_random_datetime_between_ok():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)

    with patch("src.database.simulate_client.random.randint", return_value=0):
        result = random_datetime_between(start, end)

    assert result == start


def test_random_datetime_between_raises():
    start = datetime(2026, 1, 2, tzinfo=UTC)
    end = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValueError, match="postérieure"):
        random_datetime_between(start, end)


def test_maybe_corrupted_no_error():
    profile = {"AMT_CREDIT": 1000.0, "DAYS_BIRTH": -100}

    with patch("src.database.simulate_client.random.random", return_value=1.0):
        result = maybe_corrupted_profile(profile, index=0, error_rate=0.5)

    assert result == profile


@pytest.mark.parametrize(
    ("random_values", "choice_value", "expected_key", "expected_value"),
    [
        ([0.1, 0], "empty", None, None),
        ([0.1, 1], "invalid_value", "AMT_CREDIT", "1000.0"),
        ([0.1, 2], "invalid_feature", "invalid_feature", 100),
    ],
)
def test_maybe_corrupted_error_types(
    random_values,
    choice_value,
    expected_key,
    expected_value,
):
    profile = {"AMT_CREDIT": 1000.0, "DAYS_BIRTH": -100}

    with (
        patch("src.database.simulate_client.random.random", side_effect=random_values),
        patch(
            "src.database.simulate_client.random.choice",
            side_effect=[choice_value, "AMT_CREDIT"],
        ),
        patch("builtins.print"),
    ):
        result = maybe_corrupted_profile(profile, index=1, error_rate=0.5)

    if expected_key is None:
        assert result == {}
    elif expected_key == "AMT_CREDIT":
        assert result["AMT_CREDIT"] == expected_value
    else:
        assert result[expected_key] == expected_value


def test_simulate_profils(tmp_path, monkeypatch):
    from src.database.simulate_client import simulate_profils

    data_dir = tmp_path / "data" / "original"
    schema_dir = tmp_path / "data" / "schema"
    data_dir.mkdir(parents=True)
    schema_dir.mkdir(parents=True)

    schema_path = schema_dir / "typeAdapters.json"
    schema_path.write_text(
        (BASE_DIR / "data" / "schema" / "typeAdapters.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    training_csv = data_dir / "training_data.csv"
    demo = pd.read_csv(DATA_PATH, sep=";")
    demo.to_csv(training_csv, index=False)

    with patch("builtins.print"):
        simulate_profils(tmp_path, NB_PROFILS=3, ERROR_RATE=0.0)

    profils_path = tmp_path / "data" / "profils.json"
    assert profils_path.is_file()
    profils = pd.read_json(profils_path, typ="series")
    assert len(profils) == 3
