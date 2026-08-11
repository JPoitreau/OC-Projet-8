"""Page Vue d'ensemble."""

from __future__ import annotations

import plotly.express as px
import streamlit as st


def _fmt_rate(rate: float) -> str:
    return f"{rate * 100:.1f} %"


def render() -> None:
    data = st.session_state.monitoring
    kpis = data["kpis"]
    daily = data["daily"]

    st.title("Vue d'ensemble")
    st.caption(
        f"Période : {data['start'].isoformat()} → {data['end'].isoformat()} "
        "(axe temporel : event_time, sinon requested_at)"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Requêtes totales", kpis["total"])
    col2.metric("Succès", kpis["success"])
    col3.metric("Erreurs", kpis["errors"])
    col4.metric("Taux d'erreur", _fmt_rate(kpis["error_rate"]))

    if data["logs"].empty:
        st.info("Aucune requête sur la période sélectionnée.")
        return

    left, right = st.columns(2)

    with left:
        fig_volume = px.bar(
            daily,
            x="day",
            y="total",
            title="Nombre de requêtes par jour",
            labels={"day": "Jour", "total": "Requêtes"},
        )
        fig_volume.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(fig_volume, use_container_width=True)

    with right:
        fig_error = px.line(
            daily,
            x="day",
            y="error_rate",
            title="Taux d'erreur par jour",
            markers=True,
            labels={"day": "Jour", "error_rate": "Taux d'erreur"},
        )
        fig_error.update_layout(
            yaxis_tickformat=".0%",
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        st.plotly_chart(fig_error, use_container_width=True)
