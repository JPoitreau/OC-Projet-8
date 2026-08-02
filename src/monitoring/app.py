"""Point d'entrée Streamlit — monitoring model_logs."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Projet root pour les imports `src.*` quand Streamlit lance ce fichier.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.monitoring.data import (
    compute_kpis,
    daily_aggregates,
    default_date_range,
    latency_by_event_type,
    load_logs,
    prediction_distribution,
    top_error_types,
)
from src.monitoring.pages import overview, performance, predictions, volume

st.set_page_config(
    page_title="Monitoring scoring API",
    page_icon="📊",
    layout="wide",
)


def _parse_period(raw: date | tuple[date, date] | list[date]) -> tuple[date, date]:
    if isinstance(raw, date):
        return raw, raw
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        start, end = raw[0], raw[1]
        if start is None or end is None:
            return default_date_range()
        if start > end:
            start, end = end, start
        return start, end
    return default_date_range()


@st.cache_data(ttl=60, show_spinner="Chargement des logs…")
def _cached_logs(start: date, end: date, refresh_token: int):
    _ = refresh_token
    return load_logs(start, end)


with st.sidebar:
    st.header("Filtres")
    default_start, default_end = default_date_range()
    period = st.date_input(
        "Période",
        value=(default_start, default_end),
        help="Filtre sur COALESCE(event_time, requested_at).",
    )
    start_date, end_date = _parse_period(period)

    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = 0

    if st.button("Rafraîchir", use_container_width=True):
        st.session_state.refresh_token += 1
        _cached_logs.clear()

    st.caption(f"Du {start_date.isoformat()} au {end_date.isoformat()}")

logs = _cached_logs(start_date, end_date, st.session_state.refresh_token)

st.session_state.monitoring = {
    "start": start_date,
    "end": end_date,
    "logs": logs,
    "kpis": compute_kpis(logs),
    "daily": daily_aggregates(logs),
    "latency_by_type": latency_by_event_type(logs),
    "pred_dist": prediction_distribution(logs),
    "top_errors": top_error_types(logs),
}

pages = st.navigation(
    {
        "Monitoring": [
            st.Page(
                overview.render,
                title="Vue d'ensemble",
                url_path="overview",
                default=True,
            ),
            st.Page(
                volume.render,
                title="Volume & fiabilité",
                url_path="volume",
            ),
            st.Page(
                performance.render,
                title="Performance",
                url_path="performance",
            ),
            st.Page(
                predictions.render,
                title="Prédictions",
                url_path="predictions",
            ),
        ]
    }
)
pages.run()
