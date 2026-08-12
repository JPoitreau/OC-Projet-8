"""Chargement et agrégations des logs model_logs pour le dashboard."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.database.database import engine

TOP_ERRORS_N = 10


def _period_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    start_dt = datetime.combine(start, time.min)
    end_dt = datetime.combine(end, time.max.replace(microsecond=0))
    return start_dt, end_dt


def load_logs(start: date, end: date) -> pd.DataFrame:
    """Charge model_logs filtrés sur COALESCE(event_time, requested_at)."""
    start_dt, end_dt = _period_bounds(start, end)

    query = text(
        """
        SELECT
            request_id,
            COALESCE(event_time, requested_at) AS ts,
            error,
            error_message,
            pred_class,
            execution_time_ms
        FROM model_logs
        WHERE COALESCE(event_time, requested_at) >= :start_dt
          AND COALESCE(event_time, requested_at) <= :end_dt
        ORDER BY ts
        """
    )

    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"start_dt": start_dt, "end_dt": end_dt},
        )

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["error"] = df["error"].astype(bool)
    df["execution_time_ms"] = pd.to_numeric(df["execution_time_ms"], errors="coerce")
    df["day"] = df["ts"].dt.floor("D")
    return df


def compute_kpis(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "success": 0,
            "errors": 0,
            "error_rate": 0.0,
            "p50_ms": None,
            "p95_ms": None,
        }

    errors = int(df["error"].sum())
    success = total - errors
    latency = df["execution_time_ms"].dropna()

    return {
        "total": total,
        "success": success,
        "errors": errors,
        "error_rate": errors / total if total else 0.0,
        "p50_ms": float(latency.quantile(0.50)) if not latency.empty else None,
        "p95_ms": float(latency.quantile(0.95)) if not latency.empty else None,
    }


def daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Agrégations journalières pour volume, fiabilité, perf et prédictions."""
    if df.empty:
        return pd.DataFrame(
            columns=[
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
        )

    grouped = df.groupby("day", dropna=False)

    daily = grouped.agg(
        total=("request_id", "count"),
        errors=("error", "sum"),
        mean_ms=("execution_time_ms", "mean"),
        p50_ms=("execution_time_ms", lambda s: s.quantile(0.50)),
        p95_ms=("execution_time_ms", lambda s: s.quantile(0.95)),
    ).reset_index()

    daily["errors"] = daily["errors"].astype(int)
    daily["success"] = daily["total"] - daily["errors"]
    daily["error_rate"] = daily["errors"] / daily["total"]

    success_df = df.loc[~df["error"]].copy()
    if success_df.empty:
        daily["class1_rate"] = float("nan")
    else:
        class1 = (
            success_df.assign(is_class1=success_df["pred_class"] == 1)
            .groupby("day")["is_class1"]
            .mean()
            .rename("class1_rate")
        )
        daily = daily.merge(class1, on="day", how="left")

    return daily.sort_values("day").reset_index(drop=True)


def latency_by_event_type(df: pd.DataFrame) -> pd.DataFrame:
    """Temps moyen de traitement par type d'événement (succès vs erreur)."""
    if df.empty:
        return pd.DataFrame(columns=["event_type", "mean_ms", "count"])

    summary = (
        df.assign(event_type=df["error"].map({False: "Succès", True: "Erreur"}))
        .groupby("event_type", as_index=False)
        .agg(mean_ms=("execution_time_ms", "mean"), count=("request_id", "count"))
    )
    order = {"Succès": 0, "Erreur": 1}
    return summary.sort_values("event_type", key=lambda s: s.map(order)).reset_index(
        drop=True
    )


def prediction_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Répartition des pred_class parmi les requêtes réussies."""
    success = df.loc[~df["error"]].copy() if not df.empty else df
    if success.empty:
        return pd.DataFrame(columns=["pred_class", "count"])

    dist = (
        success["pred_class"]
        .fillna(-1)
        .astype(int)
        .value_counts()
        .rename_axis("pred_class")
        .reset_index(name="count")
        .sort_values("pred_class")
        .reset_index(drop=True)
    )
    return dist


def normalize_error_type(message: Any) -> str:
    """Normalise un message d'erreur en type lisible pour le top N."""
    if message is None or (isinstance(message, float) and pd.isna(message)):
        return "Message manquant"

    text_msg = message if isinstance(message, str) else str(message)
    text_msg = text_msg.strip()

    if not text_msg:
        return "Message vide"
    if (
        "No features value was recieved" in text_msg
        or "Aucune valeur utilisateur" in text_msg
        or text_msg == "{}"
    ):
        return "Aucune feature reçue"
    if "Validation error" in text_msg or "Erreurs de validation" in text_msg:
        return "Erreur de validation"
    if (
        "Unknown feature" in text_msg
        or "Variable inconnue" in text_msg
    ):
        return "Feature inconnue"
    if "No feature selected" in text_msg or "Aucune feature sélectionnée" in text_msg:
        return "Aucune feature sélectionnée"
    if "An error occured" in text_msg or "Une erreur est survenue" in text_msg:
        return "Erreur d'inférence"

    truncated = text_msg.split("\n", maxsplit=1)[0]
    return truncated[:80] + ("…" if len(truncated) > 80 else "")


def top_error_types(df: pd.DataFrame, n: int = TOP_ERRORS_N) -> pd.DataFrame:
    errors = df.loc[df["error"]].copy() if not df.empty else df
    if errors.empty:
        return pd.DataFrame(columns=["error_type", "count"])

    errors["error_type"] = errors["error_message"].map(normalize_error_type)
    top = (
        errors["error_type"]
        .value_counts()
        .head(n)
        .rename_axis("error_type")
        .reset_index(name="count")
    )
    return top


def default_date_range() -> tuple[date, date]:
    """Période par défaut : min → max des timestamps en base, sinon 30 jours."""
    query = text(
        """
        SELECT
            MIN(COALESCE(event_time, requested_at)) AS min_ts,
            MAX(COALESCE(event_time, requested_at)) AS max_ts
        FROM model_logs
        """
    )

    with engine.connect() as connection:
        row = connection.execute(query).one()

    if row.min_ts is None or row.max_ts is None:
        end = datetime.now(tz=UTC).date()
        start = end - timedelta(days=29)
        return start, end

    start = pd.Timestamp(row.min_ts).date()
    end = pd.Timestamp(row.max_ts).date()
    return start, end
