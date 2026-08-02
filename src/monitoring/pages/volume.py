"""Page Volume & fiabilité."""

from __future__ import annotations

import plotly.express as px
import streamlit as st


def render() -> None:
    data = st.session_state.monitoring
    daily = data["daily"]
    top_errors = data["top_errors"]

    st.title("Volume & fiabilité")
    st.caption(
        f"Période : {data['start'].isoformat()} → {data['end'].isoformat()}"
    )

    if data["logs"].empty:
        st.info("Aucune requête sur la période sélectionnée.")
        return

    stacked = daily.melt(
        id_vars=["day"],
        value_vars=["success", "errors"],
        var_name="statut",
        value_name="count",
    )
    stacked["statut"] = stacked["statut"].map(
        {"success": "Succès", "errors": "Erreurs"}
    )

    fig_stacked = px.bar(
        stacked,
        x="day",
        y="count",
        color="statut",
        title="Requêtes par jour (succès / erreurs)",
        labels={"day": "Jour", "count": "Requêtes", "statut": "Statut"},
        barmode="stack",
        color_discrete_map={"Succès": "#2ca02c", "Erreurs": "#d62728"},
    )
    fig_stacked.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
    st.plotly_chart(fig_stacked, use_container_width=True)

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

    st.subheader("Top types d'erreurs")
    if top_errors.empty:
        st.success("Aucune erreur sur la période.")
    else:
        fig_top = px.bar(
            top_errors.sort_values("count"),
            x="count",
            y="error_type",
            orientation="h",
            title="Top types d'erreurs",
            labels={"count": "Occurrences", "error_type": "Type d'erreur"},
        )
        fig_top.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(fig_top, use_container_width=True)
