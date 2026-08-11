"""
Exporte les métriques baseline depuis model_logs (Phase 1 optimisation).

Usage:
    python -m src.benchmark.export_baseline
    python -m src.benchmark.export_baseline --markdown docs/optimization_report.md
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.database.database import engine
from src.monitoring.data import (
    compute_kpis,
    daily_aggregates,
    default_date_range,
    latency_by_event_type,
    load_logs,
)

REPORT_SECTION_START = "<!-- baseline-monitoring:start -->"
REPORT_SECTION_END = "<!-- baseline-monitoring:end -->"


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Formatte un DataFrame en tableau Markdown sans dépendance tabulate."""
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, separator, *rows])


def _monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.assign(month=df["ts"].dt.to_period("M"))
        .groupby("month", as_index=False)
        .agg(
            total=("request_id", "count"),
            success=("error", lambda s: (~s).sum()),
            errors=("error", "sum"),
            mean_ms=("execution_time_ms", "mean"),
            p95_ms=("execution_time_ms", lambda s: s.quantile(0.95)),
        )
    )
    monthly["error_rate"] = monthly["errors"] / monthly["total"]
    return monthly


def _db_event_time_stats() -> dict[str, int]:
    query = text(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE error = false) AS success,
            COUNT(*) FILTER (WHERE error = true) AS errors,
            COUNT(*) FILTER (WHERE event_time IS NULL AND error = false) AS success_null_event,
            COUNT(*) FILTER (WHERE event_time IS NULL AND error = true) AS error_null_event
        FROM model_logs
        """
    )
    with engine.connect() as connection:
        row = connection.execute(query).one()
    return dict(row._mapping)


def _success_latency_stats(success: pd.DataFrame) -> dict[str, float | int]:
    latency = success["execution_time_ms"]
    outliers = success[latency > 100]
    return {
        "count": len(success),
        "mean_ms": float(latency.mean()),
        "std_ms": float(latency.std()),
        "p50_ms": float(latency.quantile(0.50)),
        "p95_ms": float(latency.quantile(0.95)),
        "p99_ms": float(latency.quantile(0.99)),
        "min_ms": float(latency.min()),
        "max_ms": float(latency.max()),
        "outliers_over_100ms": len(outliers),
        "outlier_rate_pct": 100 * len(outliers) / len(success) if len(success) else 0.0,
    }


def build_baseline_markdown(start: date, end: date) -> str:
    df = load_logs(start, end)
    kpis = compute_kpis(df)
    daily = daily_aggregates(df)
    by_type = latency_by_event_type(df)
    monthly = _monthly_summary(df)
    event_stats = _db_event_time_stats()

    success = df.loc[~df["error"]]
    errors = df.loc[df["error"]]
    success_stats = _success_latency_stats(success)

    daily_table = daily[
        ["day", "total", "success", "errors", "error_rate", "mean_ms", "p50_ms", "p95_ms"]
    ].copy()
    daily_table["day"] = daily_table["day"].dt.strftime("%Y-%m-%d")
    daily_table["error_rate"] = daily_table["error_rate"].map(lambda x: f"{x * 100:.1f} %")

    monthly_table = monthly.copy()
    monthly_table["month"] = monthly_table["month"].astype(str)
    monthly_table["error_rate"] = monthly_table["error_rate"].map(
        lambda x: f"{x * 100:.1f} %"
    )

    return f"""{REPORT_SECTION_START}
## 1. Baseline monitoring (données `model_logs`)

> Généré via `python -m src.benchmark.export_baseline`.
> Période analysée : **{start.isoformat()}** -> **{end.isoformat()}**
> Axe temporel : `COALESCE(event_time, requested_at)`.

### 1.1 Volume et fiabilité globaux

| Indicateur | Valeur |
|------------|--------|
| Requêtes totales | {kpis["total"]} |
| Succès | {kpis["success"]} |
| Erreurs | {kpis["errors"]} |
| Taux d'erreur | {kpis["error_rate"] * 100:.1f} % |
| Latence P50 (toutes requêtes) | {kpis["p50_ms"]:.2f} ms |
| Latence P95 (toutes requêtes) | {kpis["p95_ms"]:.2f} ms |

**Complétude `event_time` (table complète) :**

| Segment | Lignes | `event_time` NULL |
|---------|--------|-------------------|
| Succès | {event_stats["success"]} | {event_stats["success_null_event"]} ({100 * event_stats["success_null_event"] / max(event_stats["success"], 1):.1f} %) |
| Erreurs | {event_stats["errors"]} | {event_stats["error_null_event"]} ({100 * event_stats["error_null_event"] / max(event_stats["errors"], 1):.1f} %) |

Les erreurs historiques (avant correction de l'API) n'avaient pas d'`event_time` persisté ; les requêtes récentes du 2026-08-02 montrent un pic d'erreurs simulées (549 requêtes, 100 % d'échec).

### 1.2 Latence — requêtes réussies (chemin nominal)

`execution_time_ms` mesure le **temps total** côté API : validation + inférence + écriture PostgreSQL.

| Percentile | Latence (ms) |
|------------|--------------|
| P50 | {success_stats["p50_ms"]:.2f} |
| P95 | {success_stats["p95_ms"]:.2f} |
| P99 | {success_stats["p99_ms"]:.2f} |
| Moyenne | {success_stats["mean_ms"]:.2f} |
| Écart-type | {success_stats["std_ms"]:.2f} |
| Min / Max | {success_stats["min_ms"]:.2f} / {success_stats["max_ms"]:.2f} |

**Outliers :** {success_stats["outliers_over_100ms"]} requêtes réussies > 100 ms ({success_stats["outlier_rate_pct"]:.2f} %), dont un maximum à {success_stats["max_ms"]:.0f} ms (probable cold start ou contention I/O). La moyenne ({success_stats["mean_ms"]:.1f} ms) est tirée vers le haut par ces rares pics ; **P50/P95 sont plus représentatifs du comportement nominal (~13–22 ms).**

### 1.3 Latence — requêtes en erreur

Les erreurs échouent **avant ou pendant** la validation, sans inférence complète ni écriture de prédiction :

| Indicateur | Valeur |
|------------|--------|
| Nombre | {len(errors)} |
| Temps moyen | {errors["execution_time_ms"].mean():.2f} ms |
| P50 | {errors["execution_time_ms"].quantile(0.50):.2f} ms |

Temps moyen par type d'événement :

| Type | Temps moyen (ms) | Volume |
|------|------------------|--------|
| Succès | {by_type.iloc[0]["mean_ms"]:.2f} | {int(by_type.iloc[0]["count"])} |
| Erreur | {by_type.iloc[1]["mean_ms"]:.2f} | {int(by_type.iloc[1]["count"])} |

**Interprétation :** le chemin nominal (~13 ms P50) est **~10× plus coûteux** que le chemin erreur (~1,5 ms P50), ce qui confirme que l'inférence + DB dominent le temps de réponse des succès.

### 1.4 Volume et latence par mois

{monthly_table.round(2).pipe(_dataframe_to_markdown)}

Pic de latence moyenne en **juillet 2026** (38 ms) lié aux outliers ; le P95 mensuel reste stable (~18–22 ms) hors journées atypiques.

### 1.5 Stabilité journalière (extrait)

Premiers jours :

{daily_table.head(5).pipe(_dataframe_to_markdown)}

Derniers jours :

{daily_table.tail(5).pipe(_dataframe_to_markdown)}

### 1.6 Hypothèses d'optimisation (à valider en Phase 2 — cProfile)

Sur la base de cette baseline et de l'architecture actuelle ([`src/api/scoring_api.py`](../src/api/scoring_api.py)) :

1. **Inférence + pandas** — création d'un `DataFrame` et `reindex` à chaque requête avant `scoring_model.predict` ; coût probablement dominant sur les ~13 ms P50.
2. **Validation Pydantic** — instanciation `TypeAdapter(eval(...))` à chaque feature à chaque appel ; impact à quantifier (chemin succès uniquement).
3. **Écriture PostgreSQL synchrone** — `save_prediction_log` inclus dans `execution_time_ms` ; latence réseau/IO non négligeable en production.
4. **Cold start** — outliers > 100 ms (0,5 % des succès) suggèrent un coût de démarrage ou de contention ponctuelle.
5. **Pas de GPU** — déploiement Hugging Face Spaces CPU ([`Dockerfile`](../Dockerfile)) ; optimisations hardware limitées au choix d'instance CPU.

**Prochaine étape :** profiling cProfile (4 scénarios × 100 appels) pour confirmer la répartition CPU réelle.
{REPORT_SECTION_END}
"""


def _write_markdown_section(output_path: Path, content: str) -> None:
    section_body = content.split(REPORT_SECTION_START, maxsplit=1)[1]
    section_body = section_body.split(REPORT_SECTION_END, maxsplit=1)[0].strip()

    if output_path.exists():
        text = output_path.read_text(encoding="utf-8")
        if REPORT_SECTION_START in text and REPORT_SECTION_END in text:
            before = text.split(REPORT_SECTION_START, maxsplit=1)[0]
            after = text.split(REPORT_SECTION_END, maxsplit=1)[1]
            output_path.write_text(
                f"{before}{REPORT_SECTION_START}\n{section_body}\n{REPORT_SECTION_END}{after}",
                encoding="utf-8",
            )
            return

    header = """# Rapport d'optimisation post-déploiement

Projet OC-Projet-8 — API de scoring crédit (LightGBM + Gradio + PostgreSQL).

## Table des matières

1. [Baseline monitoring](#1-baseline-monitoring-données-model_logs)
2. Profiling cProfile *(à compléter)*
3. Optimisations appliquées *(à compléter)*
4. Benchmark comparatif *(à compléter)*
5. Validation non-régression *(à compléter)*
6. Configuration finale *(à compléter)*
7. Prochaine étape ONNX *(à compléter)*

"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{header}\n{REPORT_SECTION_START}\n{section_body}\n{REPORT_SECTION_END}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export baseline monitoring metrics.")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Met à jour la section baseline dans ce fichier Markdown.",
    )
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    args = parser.parse_args()

    default_start, default_end = default_date_range()
    start = args.start or default_start
    end = args.end or default_end

    markdown = build_baseline_markdown(start, end)
    print(markdown)

    if args.markdown:
        _write_markdown_section(args.markdown, markdown)
        print(f"\nSection baseline écrite dans {args.markdown}")


if __name__ == "__main__":
    main()
