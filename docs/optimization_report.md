# Rapport d'optimisation post-déploiement

Projet OC-Projet-8 — API de scoring crédit (LightGBM + Gradio + PostgreSQL).

## Table des matières

1. [Baseline monitoring](#1-baseline-monitoring-données-model_logs)
2. [Profiling cProfile](#2-profiling-cprofile)
3. Optimisations appliquées *(à compléter)*
4. Benchmark comparatif *(à compléter)*
5. Validation non-régression *(à compléter)*
6. Configuration finale *(à compléter)*
7. Prochaine étape ONNX *(à compléter)*

<!-- baseline-monitoring:start -->
## 1. Baseline monitoring (données `model_logs`)

> Généré via `python -m src.benchmark.export_baseline`.
> Période analysée : **2026-01-01** -> **2026-08-02**
> Axe temporel : `COALESCE(event_time, requested_at)`.

### 1.1 Volume et fiabilité globaux

| Indicateur | Valeur |
|------------|--------|
| Requêtes totales | 5052 |
| Succès | 3901 |
| Erreurs | 1151 |
| Taux d'erreur | 22.8 % |
| Latence P50 (toutes requêtes) | 12.86 ms |
| Latence P95 (toutes requêtes) | 20.20 ms |

**Complétude `event_time` (table complète) :**

| Segment | Lignes | `event_time` NULL |
|---------|--------|-------------------|
| Succès | 3901 | 33 (0.8 %) |
| Erreurs | 1151 | 627 (54.5 %) |

Les erreurs historiques (avant correction de l'API) n'avaient pas d'`event_time` persisté ; les requêtes récentes du 2026-08-02 montrent un pic d'erreurs simulées (549 requêtes, 100 % d'échec).

### 1.2 Latence — requêtes réussies (chemin nominal)

`execution_time_ms` mesure le **temps total** côté API : validation + inférence + écriture PostgreSQL.

| Percentile | Latence (ms) |
|------------|--------------|
| P50 | 13.49 |
| P95 | 21.78 |
| P99 | 42.34 |
| Moyenne | 21.18 |
| Écart-type | 115.55 |
| Min / Max | 10.75 / 2483.96 |

**Outliers :** 21 requêtes réussies > 100 ms (0.54 %), dont un maximum à 2484 ms (probable cold start ou contention I/O). La moyenne (21.2 ms) est tirée vers le haut par ces rares pics ; **P50/P95 sont plus représentatifs du comportement nominal (~13–22 ms).**

### 1.3 Latence — requêtes en erreur

Les erreurs échouent **avant ou pendant** la validation, sans inférence complète ni écriture de prédiction :

| Indicateur | Valeur |
|------------|--------|
| Nombre | 1151 |
| Temps moyen | 1.13 ms |
| P50 | 1.54 ms |

Temps moyen par type d'événement :

| Type | Temps moyen (ms) | Volume |
|------|------------------|--------|
| Succès | 21.18 | 3901 |
| Erreur | 1.13 | 1151 |

**Interprétation :** le chemin nominal (~13 ms P50) est **~10× plus coûteux** que le chemin erreur (~1,5 ms P50), ce qui confirme que l'inférence + DB dominent le temps de réponse des succès.

### 1.4 Volume et latence par mois

| month | total | success | errors | mean_ms | p95_ms | error_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-01 | 622 | 548 | 74 | 12.94 | 20.3 | 11.9 % |
| 2026-02 | 596 | 524 | 72 | 13.66 | 20.97 | 12.1 % |
| 2026-03 | 644 | 561 | 83 | 16.12 | 18.54 | 12.9 % |
| 2026-04 | 654 | 580 | 74 | 15.98 | 20.99 | 11.3 % |
| 2026-05 | 618 | 548 | 70 | 14.69 | 22.53 | 11.3 % |
| 2026-06 | 610 | 537 | 73 | 13.46 | 20.71 | 12.0 % |
| 2026-07 | 759 | 603 | 156 | 38.16 | 21.64 | 20.6 % |
| 2026-08 | 549 | 0 | 549 | 1.21 | 2.93 | 100.0 % |

Pic de latence moyenne en **juillet 2026** (38 ms) lié aux outliers ; le P95 mensuel reste stable (~18–22 ms) hors journées atypiques.

### 1.5 Stabilité journalière (extrait)

Premiers jours :

| day | total | success | errors | error_rate | mean_ms | p50_ms | p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-01 | 14 | 13 | 1 | 7.1 % | 12.505357142857141 | 13.506499999999999 | 15.06405 |
| 2026-01-02 | 18 | 16 | 2 | 11.1 % | 13.582222222222223 | 13.6465 | 22.393449999999987 |
| 2026-01-03 | 27 | 20 | 7 | 25.9 % | 10.72725925925926 | 12.262 | 17.2594 |
| 2026-01-04 | 20 | 18 | 2 | 10.0 % | 12.7501 | 13.347 | 17.39095 |
| 2026-01-05 | 16 | 16 | 0 | 0.0 % | 13.00475 | 12.615 | 16.7195 |

Derniers jours :

| day | total | success | errors | error_rate | mean_ms | p50_ms | p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-28 | 16 | 15 | 1 | 6.2 % | 13.5171875 | 14.0055 | 17.66975 |
| 2026-07-29 | 14 | 12 | 2 | 14.3 % | 13.0035 | 13.1985 | 23.2692 |
| 2026-07-30 | 20 | 18 | 2 | 10.0 % | 13.757650000000002 | 13.6505 | 19.952550000000006 |
| 2026-07-31 | 21 | 18 | 3 | 14.3 % | 11.669285714285715 | 12.911 | 16.116 |
| 2026-08-02 | 549 | 0 | 549 | 100.0 % | 1.2067850637522768 | 1.761 | 2.9262000000000006 |

### 1.6 Hypothèses d'optimisation (à valider en Phase 2 — cProfile)

Sur la base de cette baseline et de l'architecture actuelle ([`src/api/scoring_api.py`](../src/api/scoring_api.py)) :

1. **Inférence + pandas** — création d'un `DataFrame` et `reindex` à chaque requête avant `scoring_model.predict` ; coût probablement dominant sur les ~13 ms P50.
2. **Validation Pydantic** — instanciation `TypeAdapter(eval(...))` à chaque feature à chaque appel ; impact à quantifier (chemin succès uniquement).
3. **Écriture PostgreSQL synchrone** — `save_prediction_log` inclus dans `execution_time_ms` ; latence réseau/IO non négligeable en production.
4. **Cold start** — outliers > 100 ms (0,5 % des succès) suggèrent un coût de démarrage ou de contention ponctuelle.
5. **Pas de GPU** — déploiement Hugging Face Spaces CPU ([`Dockerfile`](../Dockerfile)) ; optimisations hardware limitées au choix d'instance CPU.

**Prochaine étape :** profiling cProfile (4 scénarios × 100 appels) pour confirmer la répartition CPU réelle.
<!-- baseline-monitoring:end -->

<!-- profile-cprofile:start -->
## 2. Profiling cProfile

> Généré via `python -m src.benchmark.profile_api`.
> Protocole : **5 warm-up** + **100 appels** profilés par scénario (1 session cProfile par scénario).
> Échantillon : 10 profils valides cyclés sur 100 itérations.

### 2.1 Synthèse par scénario

| scenario | description | calls | total_cpu_s | avg_ms | stats_file |
| --- | --- | --- | --- | --- | --- |
| A | Validation Pydantic | 100 | 0.5 | 5.0 | profile_scenario_A.stats |
| B | Inférence pure | 100 | 6.098 | 60.98 | profile_scenario_B.stats |
| C | Inférence sans DB | 100 | 6.929 | 69.29 | profile_scenario_C.stats |
| D | Chemin API complet | 100 | 7.662 | 76.62 | profile_scenario_D.stats |

### 2.2 Top fonctions par scénario

#### Scénario A — Validation Pydantic

`validate_params` seul (sans inférence ni DB).

| ncalls | tottime_s | cumtime_s | function |
| --- | --- | --- | --- |
| 100 | 0.0086 | 0.4349 | scoring_api.py:52(validate_params) |
| 860 | 0.0128 | 0.1722 | type_adapter.py:196(__init__) |
| 960 | 0.0071 | 0.169 | frame.py:1514(iterrows) |
| 860 | 0.0177 | 0.141 | type_adapter.py:263(_init_core_attrs) |
| 860 | 0.0185 | 0.1353 | series.py:392(__init__) |
| 1160 | 0.012 | 0.0669 | construction.py:517(sanitize_array) |
| 100 | 0.0028 | 0.0623 | frame.py:698(__init__) |
| 100 | 0.0017 | 0.057 | construction.py:423(dict_to_mgr) |
| 2580 | 0.0086 | 0.0474 | series.py:1107(__getitem__) |
| 860 | 0.006 | 0.0404 | _generate_schema.py:717(generate_schema) |
| 100 | 0.0011 | 0.0324 | construction.py:96(arrays_to_mgr) |
| 2580 | 0.0083 | 0.0319 | series.py:1232(_get_value) |
| 860 | 0.0312 | 0.0319 | ~:0(<built-in method builtins.eval>) |
| 1060 | 0.0191 | 0.0309 | cast.py:1166(maybe_infer_to_datetimelike) |
| 860 | 0.0148 | 0.0305 | _schema_validator.py:22(create_schema_validator) |

#### Scénario B — Inférence pure

`scoring_model.predict` sur un vecteur pandas préparé.

| ncalls | tottime_s | cumtime_s | function |
| --- | --- | --- | --- |
| 100 | 0.0052 | 5.679 | _classification_threshold.py:828(predict) |
| 100 | 0.0024 | 5.54 | _response.py:249(_get_response_values_binary) |
| 100 | 0.0054 | 5.396 | _response.py:116(_get_response_values) |
| 100 | 0.0098 | 5.1946 | pipeline.py:800(predict_proba) |
| 400 | 0.006 | 4.2283 | _set_output.py:314(wrapped) |
| 100 | 0.3209 | 2.4876 | _data.py:3383(transform) |
| 500 | 0.0177 | 2.0694 | validation.py:2845(validate_data) |
| 26200 | 1.4707 | 1.747 | _data.py:3480(_yeo_johnson_transform) |
| 700 | 0.1026 | 1.7362 | validation.py:734(check_array) |
| 100 | 0.0171 | 1.663 | _base.py:609(transform) |
| 100 | 0.005 | 1.6035 | _base.py:335(_validate_input) |
| 100 | 0.0087 | 0.8411 | sklearn.py:1615(predict_proba) |
| 100 | 0.0114 | 0.8232 | sklearn.py:1093(predict) |
| 600 | 0.013 | 0.7827 | ~:0(<built-in method builtins.any>) |
| 26300 | 0.0251 | 0.7679 | validation.py:925(<genexpr>) |

#### Scénario C — Inférence sans DB

`infer_from_new_vector` avec écriture PostgreSQL mockée.

| ncalls | tottime_s | cumtime_s | function |
| --- | --- | --- | --- |
| 100 | 0.0127 | 6.6297 | scoring_api.py:89(infer_from_new_vector) |
| 100 | 0.0056 | 6.0995 | _classification_threshold.py:828(predict) |
| 100 | 0.0031 | 5.9503 | _response.py:249(_get_response_values_binary) |
| 100 | 0.0065 | 5.7944 | _response.py:116(_get_response_values) |
| 100 | 0.0103 | 5.5568 | pipeline.py:800(predict_proba) |
| 400 | 0.0063 | 4.517 | _set_output.py:314(wrapped) |
| 100 | 0.3362 | 2.6594 | _data.py:3383(transform) |
| 500 | 0.0168 | 2.1911 | validation.py:2845(validate_data) |
| 26200 | 1.56 | 1.8643 | _data.py:3480(_yeo_johnson_transform) |
| 700 | 0.1042 | 1.8461 | validation.py:734(check_array) |
| 100 | 0.0174 | 1.7675 | _base.py:609(transform) |
| 100 | 0.0092 | 1.7065 | _base.py:335(_validate_input) |
| 100 | 0.0107 | 0.8895 | sklearn.py:1615(predict_proba) |
| 100 | 0.012 | 0.8657 | sklearn.py:1093(predict) |
| 600 | 0.0148 | 0.822 | ~:0(<built-in method builtins.any>) |

#### Scénario D — Chemin API complet

`process_scoring_request` (validation + inférence + DB).

| ncalls | tottime_s | cumtime_s | function |
| --- | --- | --- | --- |
| 100 | 0.0124 | 7.7149 | scoring_api.py:146(process_scoring_request) |
| 100 | 0.0185 | 6.7001 | scoring_api.py:89(infer_from_new_vector) |
| 100 | 0.0052 | 5.647 | _classification_threshold.py:828(predict) |
| 100 | 0.0027 | 5.5047 | _response.py:249(_get_response_values_binary) |
| 100 | 0.006 | 5.3651 | _response.py:116(_get_response_values) |
| 100 | 0.01 | 5.1728 | pipeline.py:800(predict_proba) |
| 400 | 0.0064 | 4.1789 | _set_output.py:314(wrapped) |
| 100 | 0.3072 | 2.4266 | _data.py:3383(transform) |
| 500 | 0.0171 | 2.1017 | validation.py:2845(validate_data) |
| 700 | 0.1041 | 1.764 | validation.py:734(check_array) |
| 26200 | 1.4126 | 1.6892 | _data.py:3480(_yeo_johnson_transform) |
| 100 | 0.016 | 1.6733 | _base.py:609(transform) |
| 100 | 0.005 | 1.6185 | _base.py:335(_validate_input) |
| 100 | 0.0104 | 0.8625 | sklearn.py:1615(predict_proba) |
| 100 | 0.0176 | 0.8527 | scoring_api.py:52(validate_params) |

### 2.3 Goulots d'étranglement identifiés

1. Temps CPU moyen par appel : validation **5.00 ms**, inférence pure **60.98 ms**, inférence sans DB **69.29 ms**, chemin complet **76.62 ms**.
2. Sur le chemin complet, la validation représente ~**7 %** du temps CPU mesuré, l'inférence pure ~**80 %** ; l'écriture DB est estimée à ~**7.33 ms**/appel (delta scénario D - C).
3. **Validation (A)** : coût dominé par la création répétée de `TypeAdapter` (860 instanciations / 100 appels) et `DataFrame.iterrows` dans `validate_params`.
4. **Inférence (B/C/D)** : le pipeline sklearn/imblearn domine — `predict` / `predict_proba`, prétraitement Yeo-Johnson (`_data.py:transform`), imputation et validations numpy.
5. **Base de données (D vs C)** : `save_prediction_log` (INSERT SQLAlchemy) ajoute ~**7 ms** CPU par requête.
6. Les temps CPU cProfile (~60–80 ms/appel) sont supérieurs à la latence murale P50 du monitoring (~13 ms) : le profiler ajoute un overhead, et `execution_time_ms` en base mesure du wall-clock incluant I/O asynchrone possible ; l'analyse comparative reste valide.
7. Scénario A — fonction applicative la plus coûteuse : `scoring_api.py:52(validate_params)` (0.435 s cumulées / 100 appels).
8. Scénario B — fonction applicative la plus coûteuse : `_classification_threshold.py:828(predict)` (5.679 s cumulées / 100 appels).
9. Scénario C — fonction applicative la plus coûteuse : `scoring_api.py:89(infer_from_new_vector)` (6.630 s cumulées / 100 appels).
10. Scénario D — fonction applicative la plus coûteuse : `scoring_api.py:146(process_scoring_request)` (7.715 s cumulées / 100 appels).

**Fichiers bruts :** `reports/profile_scenario_A.stats` à `reports/profile_scenario_D.stats` (visualisation optionnelle : `snakeviz reports/profile_scenario_D.stats`).

**Conclusion Phase 2 :** les hypothèses de la section 1 sont confirmées — l'inférence sklearn (~80 % CPU) est le goulot principal ; la validation Pydantic (~7 %) et la DB (~7 ms) sont des cibles secondaires pour la Phase 3.
<!-- profile-cprofile:end -->
