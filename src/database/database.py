import os
import datetime
from sqlalchemy import (
    URL,
    create_engine,
    text,
    MetaData,
    Table
)


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
    Récupère les entrées et sorties du modèle pour les loguer dans la table.
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