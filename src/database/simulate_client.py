from pathlib import Path

import pandas as pd
from gradio_client import Client
from gradio_client.exceptions import AppError

BASE_DIR = Path(__file__).resolve().parents[2]
PROFILS_PATH = BASE_DIR / "data" / "profils.json"

SIMULATE_PROFILS = False

if SIMULATE_PROFILS:
    import json

    DATA_PATH = BASE_DIR / "data" / "original" / "training_data.csv"
    SCHEMA_PATH = BASE_DIR / "data" / "schema" / "typeAdapters.json"
    

    training_data = pd.read_csv(DATA_PATH)

    explicative_features = pd.read_json(SCHEMA_PATH, typ='series')
    explicative_features_list = list(explicative_features.index)

    explicative_training_data = training_data.loc[:,explicative_features_list]

    profils_list = [
        {
            feature: value
            for feature in explicative_features_list
            for value in [training_data.sample(n=1)[feature].item()]
            if pd.notna(value) # Nan values are imputed by the inference pipeline 
        }
        for profil in range(10)
    ]

    PROFILS_PATH.write_text(json.dumps(profils_list))

    print("New profils registered. \n")

print("Calling API... Inferences in progress.\n")

client = Client("http://127.0.0.1:7860")

if PROFILS_PATH.is_file():

    profils = pd.read_json(PROFILS_PATH, typ='series')
    for index, profil in enumerate(profils):
        try:
            client.predict(profil, api_name="/score_client")
            print(f"Profil {index} traité.")

        except AppError:
            print(f"Profil {index} non valide.")

else:
    client.predict(
        {
            "AMT_CREDIT": 6000,
            "DAYS_BIRTH": -140,
        },
        api_name="/score_client",
    )

print("All inferences completed. Ending script.")