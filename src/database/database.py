import datetime

from sqlalchemy import URL, MetaData, Table, create_engine, text


DATABASE_NAME = "scoring_db"
DATABASE_USER = "scoring_app"
DATABASE_PASSWORD = "scoring_app_pswd"
DATABASE_HOST = "localhost"
DATABASE_PORT = 5432


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

def save_prediction_log(
    requested_params: dict,
    predicted_class: int,
    execution_time_ms: float,
) -> int:
    """
    Logue les entrées et sorties de l'API dans model_logs pour les cas de
    requêtes réussies.

    Args:
    requested_params: dictionnaire de paramètres requêtés
    predicted_class: classe prédite par le modèle
    execution_time_ms: durée d'éxecution de la requête
    """

    statement = model_logs.insert().values( #commande sqlalchemy d'insertion dans la table
        requested_at=datetime.datetime.now(datetime.timezone.utc),
        requested_params=requested_params,
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
) -> int:
    """
    Logue les entrées et sorties de l'API dans model_logs pour les cas de
    requêtes échouées.

    Args:
    requested_params: dictionnaire de paramètres requêtés
    error_message: message d'erreur renvoyé aux différentes étapes de l'API
    execution_time_ms: durée d'éxecution de la requête
    """

    statement = model_logs.insert().values(
        requested_at=datetime.datetime.now(datetime.timezone.utc),
        requested_params=requested_params,
        pred_class=None,
        execution_time_ms=execution_time_ms,
        error=True,
        error_message=error_message,
    )

    with engine.begin() as connection: #executée sur engine
        result = connection.execute(statement)

    return result.inserted_primary_key[0]


if __name__ == "__main__": 
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