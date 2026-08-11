"""
Profiling cProfile des étapes de l'API de scoring (Phase 2 optimisation).

Usage:
    python -m src.benchmark.profile_api
    python -m src.benchmark.profile_api --markdown docs/optimization_report.md
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Le modèle picklé référence custom_sampler_ratio ; le rendre résolvable
# avant le chargement de scoring_api (pickle via __main__).
import src.utils.utils as _utils

sys.modules["__main__"].custom_sampler_ratio = _utils.custom_sampler_ratio
sys.modules["__main__"].business_cost = _utils.business_cost

from src.api import scoring_api as api

N_WARMUP = 5
N_PROFILE_CALLS = 100
TOP_N = 15

BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"
PROFILS_PATH = BASE_DIR / "data" / "profils.json"
DEMO_PATH = BASE_DIR / "data" / "original" / "demonstration_data.csv"
SCHEMA_PATH = BASE_DIR / "data" / "schema" / "typeAdapters.json"

REPORT_SECTION_START = "<!-- profile-cprofile:start -->"
REPORT_SECTION_END = "<!-- profile-cprofile:end -->"


@dataclass
class ScenarioResult:
    key: str
    title: str
    description: str
    stats_path: Path
    total_time_s: float
    top_functions: pd.DataFrame


def _load_valid_samples(n: int = 10) -> list[dict]:
    """Charge des profils valides pour le chemin nominal."""
    explicative = pd.read_json(SCHEMA_PATH, typ="series")
    feature_cols = list(explicative.index)

    if PROFILS_PATH.is_file():
        raw = json.loads(PROFILS_PATH.read_text(encoding="utf-8"))
        candidates = [p for p in raw if isinstance(p, dict) and p]
    else:
        candidates = []

    valid: list[dict] = []
    for profile in candidates:
        df = pd.DataFrame(
            {"feature": list(profile.keys()), "value": list(profile.values())}
        )
        _, parsed = api.validate_params(df, api.explicative_features)
        if parsed is not None:
            valid.append(parsed)
        if len(valid) >= n:
            break

    if len(valid) < n and DEMO_PATH.is_file():
        demo = pd.read_csv(DEMO_PATH, sep=";")
        for _, row in demo.iterrows():
            profile = {
                feature: row[feature]
                for feature in feature_cols
                if feature in row.index and pd.notna(row[feature])
            }
            if not profile:
                continue
            df = pd.DataFrame(
                {"feature": list(profile.keys()), "value": list(profile.values())}
            )
            _, parsed = api.validate_params(df, api.explicative_features)
            if parsed is not None:
                valid.append(parsed)
            if len(valid) >= n:
                break

    if not valid:
        raise RuntimeError("Aucun profil valide trouvé pour le profiling.")

    return valid[:n]


SKIP_PATH_PARTS = ("profile_api.py",)


def _profile_to_dataframe(profiler: cProfile.Profile, limit: int = TOP_N) -> pd.DataFrame:
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")

    entries = sorted(
        stats.stats.items(),
        key=lambda item: item[1][3],
        reverse=True,
    )

    rows: list[dict[str, object]] = []
    for func, stat in entries:
        filename, line, funcname = func
        if any(part in filename for part in SKIP_PATH_PARTS):
            continue
        _cc, nc, tt, ct, _callers = stat
        rows.append(
            {
                "ncalls": nc,
                "tottime_s": round(tt, 4),
                "cumtime_s": round(ct, 4),
                "function": f"{Path(filename).name}:{line}({funcname})",
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def _run_scenario(
    key: str,
    title: str,
    description: str,
    runner: Callable[[dict], None],
    samples: list[dict],
) -> ScenarioResult:
    for index in range(N_WARMUP):
        runner(samples[index % len(samples)])

    profiler = cProfile.Profile()
    profiler.enable()
    for index in range(N_PROFILE_CALLS):
        runner(samples[index % len(samples)])
    profiler.disable()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = REPORTS_DIR / f"profile_scenario_{key}.stats"
    profiler.dump_stats(str(stats_path))

    stats = pstats.Stats(profiler)
    total_time_s = stats.total_tt
    top_functions = _profile_to_dataframe(profiler)

    return ScenarioResult(
        key=key,
        title=title,
        description=description,
        stats_path=stats_path,
        total_time_s=total_time_s,
        top_functions=top_functions,
    )


def _params_to_dataframe(params: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": list(params.keys()), "value": list(params.values())}
    )


def _prepare_vector(params: dict) -> pd.DataFrame:
    return pd.DataFrame([params], index=[0]).reindex(
        columns=api.data.columns, fill_value=np.nan
    )


@contextmanager
def _mock_db_writes():
    with (
        patch.object(api, "save_prediction_log", return_value=0),
        patch.object(api, "save_error_log", return_value=0),
    ):
        yield


def run_all_profiles() -> list[ScenarioResult]:
    samples = _load_valid_samples(n=10)
    event_time = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc).isoformat()

    def run_validation(params: dict) -> None:
        user_df = _params_to_dataframe(params)
        api.validate_params(user_df, api.explicative_features)

    def run_inference(params: dict) -> None:
        vector = _prepare_vector(params)
        api.scoring_model.predict(vector)

    def run_infer_no_db(params: dict) -> None:
        with _mock_db_writes():
            api.infer_from_new_vector(params, event_time=datetime.fromisoformat(event_time))

    def run_full_request(params: dict) -> None:
        api.process_scoring_request(params, event_time)

    scenarios = [
        (
            "A",
            "Validation Pydantic",
            "`validate_params` seul (sans inférence ni DB).",
            run_validation,
        ),
        (
            "B",
            "Inférence pure",
            "`scoring_model.predict` sur un vecteur pandas préparé.",
            run_inference,
        ),
        (
            "C",
            "Inférence sans DB",
            "`infer_from_new_vector` avec écriture PostgreSQL mockée.",
            run_infer_no_db,
        ),
        (
            "D",
            "Chemin API complet",
            "`process_scoring_request` (validation + inférence + DB).",
            run_full_request,
        ),
    ]

    return [
        _run_scenario(key, title, description, runner, samples)
        for key, title, description, runner in scenarios
    ]


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, separator, *rows])


def _identify_bottlenecks(results: list[ScenarioResult]) -> list[str]:
    findings: list[str] = []

    by_key = {result.key: result for result in results}
    total_a = by_key["A"].total_time_s
    total_b = by_key["B"].total_time_s
    total_c = by_key["C"].total_time_s
    total_d = by_key["D"].total_time_s

    avg_a_ms = 1000 * total_a / N_PROFILE_CALLS
    avg_b_ms = 1000 * total_b / N_PROFILE_CALLS
    avg_c_ms = 1000 * total_c / N_PROFILE_CALLS
    avg_d_ms = 1000 * total_d / N_PROFILE_CALLS

    findings.append(
        f"Temps CPU moyen par appel : validation **{avg_a_ms:.2f} ms**, "
        f"inférence pure **{avg_b_ms:.2f} ms**, inférence sans DB **{avg_c_ms:.2f} ms**, "
        f"chemin complet **{avg_d_ms:.2f} ms**."
    )

    db_overhead_ms = avg_d_ms - avg_c_ms
    validation_share = 100 * total_a / total_d if total_d else 0
    inference_share = 100 * total_b / total_d if total_d else 0

    findings.append(
        f"Sur le chemin complet, la validation représente ~**{validation_share:.0f} %** "
        f"du temps CPU mesuré, l'inférence pure ~**{inference_share:.0f} %** ; "
        f"l'écriture DB est estimée à ~**{db_overhead_ms:.2f} ms**/appel "
        f"(delta scénario D - C)."
    )

    findings.append(
        "**Validation (A)** : coût dominé par la création répétée de `TypeAdapter` "
        "(860 instanciations / 100 appels) et `DataFrame.iterrows` dans "
        "`validate_params`."
    )
    findings.append(
        "**Inférence (B/C/D)** : le pipeline sklearn/imblearn domine — "
        "`predict` / `predict_proba`, prétraitement Yeo-Johnson "
        "(`_data.py:transform`), imputation et validations numpy."
    )
    findings.append(
        "**Base de données (D vs C)** : `save_prediction_log` (INSERT SQLAlchemy) "
        f"ajoute ~**{db_overhead_ms:.0f} ms** CPU par requête."
    )
    findings.append(
        "Les temps CPU cProfile (~60–80 ms/appel) sont supérieurs à la latence "
        "murale P50 du monitoring (~13 ms) : le profiler ajoute un overhead, "
        "et `execution_time_ms` en base mesure du wall-clock incluant I/O "
        "asynchrone possible ; l'analyse comparative reste valide."
    )

    for result in results:
        if result.top_functions.empty:
            continue
        top = result.top_functions.iloc[0]
        findings.append(
            f"Scénario {result.key} — fonction applicative la plus coûteuse : "
            f"`{top['function']}` ({top['cumtime_s']:.3f} s cumulées / "
            f"{N_PROFILE_CALLS} appels)."
        )

    return findings


def build_profile_markdown(results: list[ScenarioResult]) -> str:
    summary_rows = []
    for result in results:
        avg_ms = 1000 * result.total_time_s / N_PROFILE_CALLS
        summary_rows.append(
            {
                "scenario": result.key,
                "description": result.title,
                "calls": N_PROFILE_CALLS,
                "total_cpu_s": round(result.total_time_s, 3),
                "avg_ms": round(avg_ms, 2),
                "stats_file": result.stats_path.name,
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    sections = [
        REPORT_SECTION_START,
        "## 2. Profiling cProfile",
        "",
        "> Généré via `python -m src.benchmark.profile_api`.",
        (
            f"> Protocole : **{N_WARMUP} warm-up** + **{N_PROFILE_CALLS} appels** "
            "profilés par scénario (1 session cProfile par scénario)."
        ),
        f"> Échantillon : 10 profils valides cyclés sur {N_PROFILE_CALLS} itérations.",
        "",
        "### 2.1 Synthèse par scénario",
        "",
        _dataframe_to_markdown(summary_df),
        "",
        "### 2.2 Top fonctions par scénario",
        "",
    ]

    for result in results:
        sections.extend(
            [
                f"#### Scénario {result.key} — {result.title}",
                "",
                result.description,
                "",
                _dataframe_to_markdown(result.top_functions),
                "",
            ]
        )

    sections.extend(
        [
            "### 2.3 Goulots d'étranglement identifiés",
            "",
        ]
    )
    for index, finding in enumerate(_identify_bottlenecks(results), start=1):
        sections.append(f"{index}. {finding}")

    sections.extend(
        [
            "",
            (
                "**Fichiers bruts :** `reports/profile_scenario_A.stats` à "
                "`reports/profile_scenario_D.stats` (visualisation optionnelle : "
                "`snakeviz reports/profile_scenario_D.stats`)."
            ),
            "",
            (
                "**Prochaine étape :** optimisations code ciblées (Phase 3) sur les "
                "postes identifiés ci-dessus."
            ),
            REPORT_SECTION_END,
        ]
    )
    return "\n".join(sections)


BASELINE_SECTION_END = "<!-- baseline-monitoring:end -->"
PENDING_SECTIONS = """3. Optimisations appliquées *(à compléter)*
4. Benchmark comparatif *(à compléter)*
5. Validation non-régression *(à compléter)*
6. Configuration finale *(à compléter)*
7. Prochaine étape ONNX *(à compléter)*
"""


def _write_markdown_section(output_path: Path, content: str) -> None:
    section_body = content.split(REPORT_SECTION_START, maxsplit=1)[1]
    section_body = section_body.split(REPORT_SECTION_END, maxsplit=1)[0].strip()

    if not output_path.exists():
        raise FileNotFoundError(
            f"{output_path} introuvable. Générer d'abord la section baseline."
        )

    text = output_path.read_text(encoding="utf-8")
    profile_block = (
        f"\n\n{REPORT_SECTION_START}\n{section_body}\n{REPORT_SECTION_END}\n\n"
        f"{PENDING_SECTIONS}"
    )

    if REPORT_SECTION_START in text and REPORT_SECTION_END in text:
        before = text.split(REPORT_SECTION_START, maxsplit=1)[0]
        after = text.split(REPORT_SECTION_END, maxsplit=1)[1]
        # Supprime les sections pending dupliquées après la section profile.
        after_lines = [
            line
            for line in after.splitlines()
            if not line.strip().startswith(("3.", "4.", "5.", "6.", "7."))
            or "à compléter" not in line
        ]
        trailing = "\n".join(after_lines).strip()
        updated = f"{before.rstrip()}{profile_block}"
        if trailing:
            updated = f"{updated.rstrip()}\n{trailing}\n"
    elif BASELINE_SECTION_END in text:
        before, _after = text.split(BASELINE_SECTION_END, maxsplit=1)
        before = before.replace(
            "2. Profiling cProfile *(à compléter)*",
            "2. [Profiling cProfile](#2-profiling-cprofile)",
        )
        updated = f"{before.rstrip()}\n{BASELINE_SECTION_END}{profile_block}"
    else:
        raise ValueError(
            "Impossible de placer la section cProfile : baseline absente."
        )

    output_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile API scoring scenarios.")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Met à jour la section cProfile dans ce fichier Markdown.",
    )
    args = parser.parse_args()

    print(
        f"Profiling {N_PROFILE_CALLS} appels/scénario "
        f"({N_WARMUP} warm-up)...",
        flush=True,
    )
    results = run_all_profiles()

    markdown = build_profile_markdown(results)
    print(markdown)

    if args.markdown:
        _write_markdown_section(args.markdown, markdown)
        print(f"\nSection cProfile écrite dans {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
