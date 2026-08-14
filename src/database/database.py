import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, MetaData, Table, create_engine

BASE_DIR = Path(__file__).resolve().parents[2]

# Bascule automatique vers un stockage CSV lorsqu'aucune base PostgreSQL
# n'est accessible (ex : déploiement Hugging Face Spaces, qui définit
# automatiquement la variable `SPACE_ID`). `APP_ENV=remote` permet de forcer
# ce mode manuellement (ex : pour le tester en local).
IS_REMOTE = os.getenv("APP_ENV", "local").strip().lower() == "remote" or bool(
    os.getenv("SPACE_ID")
)

# Colonnes reproduisant la structure de la table `model_logs` PostgreSQL.
REMOTE_LOGS_PATH = BASE_DIR / "data" / "model_logs_remote.csv"
LOG_COLUMNS = [
    "request_id",
    "requested_at",
    "requested_params",
    "event_time",
    "pred_class",
    "execution_time_ms",
    "error",
    "error_message",
]

DATABASE_NAME = "scoring_db"
DATABASE_USER = "scoring_app"
DATABASE_PASSWORD = "scoring_app_pswd"
DATABASE_HOST = "localhost"
DATABASE_PORT = 5432

if not IS_REMOTE:
    DATABASE_URL = URL.create(  #lien de connexion à la base de données
        drivername="postgresql+psycopg2",
        username=DATABASE_USER,
        password=DATABASE_PASSWORD,
        host=DATABASE_HOST,
        port=DATABASE_PORT,
        database=DATABASE_NAME,
    )

    #connexion
    engine = create_engine(
        DATABASE_URL
    )

    #Je récupère la table de logs avec la méthode metadata qui la lit automatiquement
    metadata = MetaData()

    model_logs = Table(
        "model_logs",
        metadata,
        autoload_with=engine,
    )


def _append_csv_log(row: dict) -> int:
    """
    Ajoute une ligne dans le csv de remplacement de `model_logs` utilisé en
    mode distant (`IS_REMOTE`). Le fichier est recréé avec l'en-tête s'il
    n'existe pas encore.

    Note : le stockage n'est pas persistant entre redéploiements sur un
    Hugging Face Space standard (système de fichiers éphémère).
    """
    if REMOTE_LOGS_PATH.exists():
        existing = pd.read_csv(REMOTE_LOGS_PATH)
    else:
        existing = pd.DataFrame(columns=LOG_COLUMNS)

    next_id = int(existing["request_id"].max()) + 1 if not existing.empty else 1
    new_row = pd.DataFrame([{"request_id": next_id, **row}], columns=LOG_COLUMNS)

    REMOTE_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated = new_row if existing.empty else pd.concat(
        [existing, new_row], ignore_index=True
    )
    updated.to_csv(REMOTE_LOGS_PATH, index=False)

    return next_id

def save_prediction_log(
    requested_params: dict,
    predicted_class: int,
    execution_time_ms: float,
    event_time: datetime | None = None,
) -> int:
    """
    Logue les entrées et sorties de l'API dans model_logs pour les cas de
    requêtes réussies.

    Args:
    requested_params: dictionnaire de paramètres requêtés
    predicted_class: classe prédite par le modèle
    execution_time_ms: durée d'éxecution de la requête
    """
    requested_at = datetime.now(UTC)

    if IS_REMOTE:
        return _append_csv_log(
            {
                "requested_at": requested_at.isoformat(),
                "requested_params": json.dumps(requested_params),
                "event_time": event_time.isoformat() if event_time else None,
                "pred_class": predicted_class,
                "execution_time_ms": execution_time_ms,
                "error": False,
                "error_message": None,
            }
        )

    statement = model_logs.insert().values( #commande sqlalchemy d'insertion dans la table
        requested_at=requested_at,
        requested_params=requested_params,
        event_time = event_time,
        pred_class=predicted_class,
        execution_time_ms=execution_time_ms,
        error=False,
        error_message=None,
    )

    with engine.begin() as connection: #executée sur engine
        result = connection.execute(statement)

    return result.inserted_primary_key[0] #revoie la clé primaire générée automatiquement dans la table => l'id de la requête

def save_error_log(
        requested_params: dict,
        error_message: str,
        execution_time_ms: float | None = None,
        event_time: datetime | None = None
) -> int:
    """
    Logue les entrées et sorties de l'API dans model_logs pour les cas de
    requêtes échouées.

    Args:
    requested_params: dictionnaire de paramètres requêtés
    error_message: message d'erreur renvoyé aux différentes étapes de l'API
    execution_time_ms: durée d'éxecution de la requête
    """
    requested_at = datetime.now(UTC)

    if IS_REMOTE:
        return _append_csv_log(
            {
                "requested_at": requested_at.isoformat(),
                "requested_params": json.dumps(requested_params),
                "event_time": event_time.isoformat() if event_time else None,
                "pred_class": None,
                "execution_time_ms": execution_time_ms,
                "error": True,
                "error_message": error_message,
            }
        )

    statement = model_logs.insert().values(
        requested_at=requested_at,
        requested_params=requested_params,
        event_time=event_time,
        pred_class=None,
        execution_time_ms=execution_time_ms,
        error=True,
        error_message=error_message,
    )

    with engine.begin() as connection: #executée sur engine
        result = connection.execute(statement)

    return result.inserted_primary_key[0]


if __name__ == "__main__":  # pragma: no cover 
#Test manuel en lançant directement le script. Empêche l'exécution si le script est importé.
    request_id = save_prediction_log(
        requested_params={
            "AMT_INCOME_TOTAL": 150000,
            "DAYS_BIRTH": -12000,
        },
        predicted_class=0,
        execution_time_ms=12.5,
    )

    print(f"Ligne créée avec request_id={request_id}")