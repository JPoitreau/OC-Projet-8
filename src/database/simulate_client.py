import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from gradio_client import Client
from gradio_client.exceptions import AppError


def random_datetime_between(
    start: datetime,
    end: datetime,
) -> datetime:
    if end <= start:
        raise ValueError(
            "La date de fin doit être postérieure à la date de début."
        )

    total_seconds = int((end - start).total_seconds())
    random_seconds = random.randint(0, total_seconds)

    return start + timedelta(seconds=random_seconds)

def maybe_corrupted_profile(
    profile: dict,
    index: int,
    error_rate: float = 0.05,
) -> dict:
    corrupted_profile = profile.copy()
    possible_errors = ["empty", "invalid_value", "invalid_feature"]

    if random.random() >= error_rate:
        return profile

    error = random.choice(possible_errors)

    if error == "empty":
        corrupted_profile = {}
    elif error == "invalid_value":
        selected_feature = random.choice(
            list(corrupted_profile.keys())
        )   
        corrupted_profile[selected_feature] = str(corrupted_profile[selected_feature])
    else:
        corrupted_profile['invalid_feature'] = 100

    print(f"Profil {index} has been corrupted with error : {error}")

    return corrupted_profile

def simulate_profils(BASE_DIR, NB_PROFILS, ERROR_RATE):
    print("Simulating profils...")
    import json

    DATA_PATH = BASE_DIR / "data" / "original" / "training_data.csv"
    SCHEMA_PATH = BASE_DIR / "data" / "schema" / "typeAdapters.json"
    PROFILS_PATH = BASE_DIR / "data" / "profils.json"
    

    training_data = pd.read_csv(DATA_PATH)

    explicative_features = pd.read_json(SCHEMA_PATH, typ='series')
    explicative_features_list = list(explicative_features.index)

    #explicative_training_data = training_data.loc[:,explicative_features_list]

    profils_list = [
        {
            feature: value
            for feature in explicative_features_list
            for value in [training_data.sample(n=1)[feature].item()]
            if pd.notna(value) # Nan values are imputed by the inference pipeline 
        }
        for profil in range(NB_PROFILS)
    ]

    for index, profil in enumerate(profils_list):
        profils_list[index] = maybe_corrupted_profile(
                                                    profil, 
                                                    index, 
                                                    ERROR_RATE
                                                    )

    PROFILS_PATH.write_text(json.dumps(profils_list))

    print("New profils registered. \n")

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[2]
    PROFILS_PATH = BASE_DIR / "data" / "profils.json"

    SIMULATE_PROFILS = True
    NB_PROFILS = 2
    ERROR_RATE = 0

    API_URL = "http://127.0.0.1:7860"
    ENDPOINT = "/score_client"

    client = Client(API_URL)

    if SIMULATE_PROFILS:
        simulate_profils(BASE_DIR, NB_PROFILS, ERROR_RATE)

    if PROFILS_PATH.is_file():

        print("Calling API... Inferences in progress.\n")

        simulation_start = datetime(
            2026, 1, 1,
            tzinfo=UTC,
        )
        simulation_end = datetime(
            2026, 7, 31, 23, 59, 59,
            tzinfo=datetime.now().astimezone().tzinfo,
        )

        profils = pd.read_json(PROFILS_PATH, typ='series')
        for index, profil in enumerate(profils):

            event_time = random_datetime_between(
            simulation_start,
            simulation_end,
            )
            try:
                client.predict(profil, 
                            event_time.isoformat(), 
                            api_name=ENDPOINT)
                print(f"Profil {index} traité.")

            except AppError:
                print(f"Profil {index} non valide.")

    else:
        event_time = datetime(
        2026, 3, 31, 23, 59, 59,
        tzinfo=UTC,
        ).isoformat()

        client.predict(
            {
                "AMT_CREDIT": 6000,
                "DAYS_BIRTH": -140,
            },
            event_time,
            api_name="/score_client",
        )

    print("All inferences completed. Ending script.")