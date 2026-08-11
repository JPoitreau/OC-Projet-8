"""Page Prédictions."""

from __future__ import annotations

import plotly.express as px
import streamlit as st


def render() -> None:
    data = st.session_state.monitoring
    pred_dist = data["pred_dist"]
    daily = data["daily"]

    st.title("Prédictions")
    st.caption(
        f"Période : {data['start'].isoformat()} → {data['end'].isoformat()} "
        "(requêtes réussies uniquement)"
    )

    success_count = data["kpis"]["success"]
    if success_count == 0:
        st.info("Aucune prédiction réussie sur la période sélectionnée.")
        return

    labels = pred_dist.copy()
    labels["label"] = labels["pred_class"].apply(
        lambda c: {0: "Classe 0", 1: "Classe 1"}.get(int(c), f"Classe {c}")
    )

    left, right = st.columns(2)

    with left:
        fig_pie = px.pie(
            labels,
            names="label",
            values="count",
            title="Répartition des prédictions",
            hole=0.35,
        )
        fig_pie.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        fig_bar = px.bar(
            labels,
            x="label",
            y="count",
            title="Volume par classe prédite",
            labels={"label": "Classe", "count": "Volume"},
            color="label",
        )
        fig_bar.update_layout(
            showlegend=False,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    class1_daily = daily.dropna(subset=["class1_rate"])
    if class1_daily.empty:
        st.info("Pas assez de succès pour tracer le taux de classe 1.")
        return

    fig_rate = px.line(
        class1_daily,
        x="day",
        y="class1_rate",
        title="Évolution du taux de classe 1",
        markers=True,
        labels={"day": "Jour", "class1_rate": "Taux de classe 1"},
    )
    fig_rate.update_layout(
        yaxis_tickformat=".0%",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    st.plotly_chart(fig_rate, use_container_width=True)
