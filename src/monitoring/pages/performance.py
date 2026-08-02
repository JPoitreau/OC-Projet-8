"""Page Performance."""

from __future__ import annotations

import plotly.express as px
import streamlit as st


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f} ms"


def render() -> None:
    data = st.session_state.monitoring
    kpis = data["kpis"]
    daily = data["daily"]
    by_type = data["latency_by_type"]

    st.title("Performance")
    st.caption(
        f"Période : {data['start'].isoformat()} → {data['end'].isoformat()}"
    )

    col1, col2 = st.columns(2)
    col1.metric("Latence P50", _fmt_ms(kpis["p50_ms"]))
    col2.metric("Latence P95", _fmt_ms(kpis["p95_ms"]))

    if data["logs"].empty:
        st.info("Aucune requête sur la période sélectionnée.")
        return

    fig_mean = px.line(
        daily,
        x="day",
        y="mean_ms",
        title="Temps moyen de traitement par jour",
        markers=True,
        labels={"day": "Jour", "mean_ms": "Temps moyen (ms)"},
    )
    fig_mean.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
    st.plotly_chart(fig_mean, use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig_type = px.bar(
            by_type,
            x="event_type",
            y="mean_ms",
            title="Temps moyen par type d'événement",
            labels={
                "event_type": "Type",
                "mean_ms": "Temps moyen (ms)",
            },
            text="count",
            color="event_type",
            color_discrete_map={"Succès": "#2ca02c", "Erreur": "#d62728"},
        )
        fig_type.update_traces(texttemplate="n=%{text}", textposition="outside")
        fig_type.update_layout(
            showlegend=False,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with right:
        fig_p95 = px.line(
            daily,
            x="day",
            y="p95_ms",
            title="Latence P95 par jour",
            markers=True,
            labels={"day": "Jour", "p95_ms": "P95 (ms)"},
        )
        fig_p95.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(fig_p95, use_container_width=True)
