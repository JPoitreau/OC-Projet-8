"""Teste les fonctions de monitoring des données. 	
Agrégations, KPIs, normalisation d'erreurs, load_logs"""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.monitoring.data import (
    compute_kpis,
    daily_aggregates,
    default_date_range,
    latency_by_event_type,
    load_logs,
    normalize_error_type,
    prediction_distribution,
    top_error_types,
)


def test_compute_kpis_empty():
    kpis = compute_kpis(pd.DataFrame())

    assert kpis == {
        "total": 0,
        "success": 0,
        "errors": 0,
        "error_rate": 0.0,
        "p50_ms": None,
        "p95_ms": None,
    }


def test_compute_kpis_mixed(logs_df_with_day):
    kpis = compute_kpis(logs_df_with_day)

    assert kpis["total"] == 5
    assert kpis["success"] == 3
    assert kpis["errors"] == 2
    assert kpis["error_rate"] == pytest.approx(0.4)
    assert kpis["p50_ms"] == pytest.approx(12.5)
    assert kpis["p95_ms"] == pytest.approx(19.25)


def test_daily_aggregates_empty():
    daily = daily_aggregates(pd.DataFrame())

    assert list(daily.columns) == [
        "day",
        "total",
        "success",
        "errors",
        "error_rate",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "class1_rate",
    ]
    assert daily.empty


def test_daily_aggregates_full(logs_df_with_day):
    daily = daily_aggregates(logs_df_with_day)

    assert len(daily) == 3
    assert daily["total"].sum() == 5
    assert daily["success"].sum() == 3
    assert daily["errors"].sum() == 2
    assert daily["class1_rate"].notna().any()


def test_daily_aggregates_no_success():
    df = pd.DataFrame(
        {
            "request_id": [1, 2],
            "ts": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "error": [True, True],
            "error_message": ["err", "err"],
            "pred_class": [None, None],
            "execution_time_ms": [1.0, 2.0],
            "day": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        }
    )

    daily = daily_aggregates(df)

    assert daily["success"].sum() == 0
    assert daily["class1_rate"].isna().all()


def test_latency_by_event_type(logs_df_with_day):
    empty = latency_by_event_type(pd.DataFrame())
    assert empty.empty

    summary = latency_by_event_type(logs_df_with_day)
    assert set(summary["event_type"]) == {"Succès", "Erreur"}
    assert summary["count"].sum() == 5


def test_prediction_distribution(logs_df_with_day):
    empty = prediction_distribution(pd.DataFrame())
    assert empty.empty

    dist = prediction_distribution(logs_df_with_day)
    assert set(dist["pred_class"]) == {0, 1}
    assert dist["count"].sum() == 3


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (None, "Message manquant"),
        ("", "Message vide"),
        ("No features value was recieved", "Aucune feature reçue"),
        ("Validation error: bad type", "Erreur de validation"),
        ("Unknown feature, not supported", "Feature inconnue"),
        ("No feature selected.", "Aucune feature sélectionnée"),
        ("An error occured: boom", "Erreur d'inférence"),
        ("x" * 100, "x" * 80 + "…"),
    ],
)
def test_normalize_error_type(message, expected):
    assert normalize_error_type(message) == expected


def test_top_error_types(logs_df_with_day):
    empty = top_error_types(pd.DataFrame())
    assert empty.empty

    top = top_error_types(logs_df_with_day, n=2)
    assert "error_type" in top.columns
    assert top["count"].sum() == 2


def test_load_logs(monkeypatch, sample_logs_df):
    monkeypatch.setattr(
        "src.monitoring.data.pd.read_sql",
        lambda *args, **kwargs: sample_logs_df.copy(),
    )

    result = load_logs(date(2026, 1, 1), date(2026, 1, 31))

    assert len(result) == 5
    assert result["error"].dtype == bool
    assert "day" in result.columns


def test_load_logs_empty(monkeypatch):
    monkeypatch.setattr(
        "src.monitoring.data.pd.read_sql",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    result = load_logs(date(2026, 1, 1), date(2026, 1, 31))
    assert result.empty


def test_default_date_range_no_data(monkeypatch):
    row = MagicMock(min_ts=None, max_ts=None)
    monkeypatch.setattr(
        "src.monitoring.data.engine.connect",
        lambda: MagicMock(
            __enter__=lambda self: MagicMock(execute=lambda *a, **k: MagicMock(one=lambda: row)),
            __exit__=lambda *a: False,
        ),
    )

    start, end = default_date_range()

    assert start <= end
    assert (end - start).days == 29


def test_default_date_range_with_data(monkeypatch):
    row = MagicMock(
        min_ts=datetime(2026, 1, 1, tzinfo=UTC),
        max_ts=datetime(2026, 2, 1, tzinfo=UTC),
    )
    monkeypatch.setattr(
        "src.monitoring.data.engine.connect",
        lambda: MagicMock(
            __enter__=lambda self: MagicMock(execute=lambda *a, **k: MagicMock(one=lambda: row)),
            __exit__=lambda *a: False,
        ),
    )

    start, end = default_date_range()

    assert start == date(2026, 1, 1)
    assert end == date(2026, 2, 1)
