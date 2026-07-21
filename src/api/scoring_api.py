#Commande de lancement du script: python -m src.api.scoring_api

import gradio as gr
import os
import pandas as pd
import numpy as np
<<<<<<< Updated upstream

data_path = "../../data/original"
data = pd.read_csv(os.path.join(data_path, 'demonstration_data.csv'))
variables_list = list(data.columns)
mean_data = data.mean()
=======
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from pickle import load
import __main__
from time import perf_counter

from src.utils.utils import custom_sampler_ratio, business_cost
from src.database.database import save_prediction_log

setattr(__main__, "custom_sampler_ratio", custom_sampler_ratio)
setattr(__main__, "business_cost", business_cost)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "original" / "demonstration_data.csv"
SCHEMA_PATH = BASE_DIR / "data" / "schema" / "typeAdapters.json"
MODEL_PATH = BASE_DIR / "src" / "model" / "lgb_model.pkl"

data = pd.read_csv(DATA_PATH, sep = ";")
features_list = list(data.columns)

explicative_features = pd.read_json(SCHEMA_PATH, typ='series')
explicative_features_list = list(explicative_features.index)

with open(MODEL_PATH, "rb") as f:
    scoring_model = load(f)
>>>>>>> Stashed changes

def update_table(features: list[str] | str | None):
    if not features:
        return pd.DataFrame(
            columns = ["variable", "value"]
        )
    if isinstance(features, str):
        features = [features] 
    return pd.DataFrame({
        "variable" : features,
        "value" : [np.nan]*len(features)
    })

<<<<<<< Updated upstream
=======
def validate_params(df: pd.DataFrame,
                    types : pd.Series):
    """
    Validate parameters type using pydantic.

    Args:
    df: 2 columns feature-value
    types: series with features as index giving awaited types
    """
    errors = []
    parsed = {}

    if df.empty:
        return "Aucun paramètre sélectionné.", None

    for _, row in df.iterrows():
        feature = row["feature"]
        value = row["value"]
        adapter = TypeAdapter(eval(types[str(feature)]))

        try:
            parsed[feature] = adapter.validate_python(value)
        except ValidationError as e:
            errors.append(
                f"- `{feature}` : {e.errors()[0]['msg']} — valeur reçue : `{value}`"
            )

    if errors:
        return "Erreurs de validation :\n\n" + "\n".join(f"- {e}" for e in errors), None

    return "✅ Paramètres valides.", parsed

def infer_from_new_vector(params: dict):
    """
    Create new vector from given parameters. Missing parameters are filled
    with nan values and imputed in the model pipeline. Save the prediction
    and the parameters into the database. Return the error message if their
    is a problème

    Args:
    params: dictionnary of features and keys with new values.
    """
    start_time = perf_counter()
    try:
        new_vector = (pd.DataFrame([params], index=[0])
                    .reindex(columns=data.columns, fill_value=np.nan))
        
        prediction = scoring_model.predict(new_vector)
        
        execution_time_ms = (perf_counter() - start_time) * 1000

        predicted_class = int(prediction[0])

        request_id = save_prediction_log(
            requested_params=params,
            predicted_class=predicted_class,
            execution_time_ms=execution_time_ms,
        )

        message = (
            f"✅ Prédiction enregistrée dans PostgreSQL. "
            f"Identifiant de requête : '{request_id}'."
        )

        return prediction.tolist(), message

    except Exception as error:
        return(
            None,
            f"Une erreur est survenue : '{error}'"
        )

def process_scoring_request(user_values: dict):
    """
    Valide les valeurs, effectue la prédiction
    et enregistre la requête dans PostgreSQL. Permet de simuler une requête
    utlisateur complète à partir d'un dictionnaire clés:valeurs
    """
    user_dataframe = pd.DataFrame(
        {
            "feature": list(user_values.keys()),
            "value": list(user_values.values()),
        }
    )

    #Appelle validate_params à partir du dataframe 
    validation_message, parsed_params = validate_params(
        user_dataframe,
        explicative_features,
    )

    #Si parsed_params est vide, une erreur de paramètre a été détectée
    if parsed_params is None:
        raise ValueError(validation_message)

    #Si tout est bon on appel la fonction d'inférence qui sauvegarde la
    #prédiction et retourne la prédiction et la confirmation d'enregistrement.
    prediction, database_message = infer_from_new_vector(parsed_params)
    #Retourne les résultats dans la console pour le débugage
    return {
        "prediction": prediction,
        "database_status": database_message,
        "validated_params": parsed_params,
    }


>>>>>>> Stashed changes
with gr.Blocks() as demo:
    choosed_features = gr.Dropdown(
        choices = variables_list,
        multiselect = True,
        value = list
    )
    
    user_table = gr.Dataframe(
        headers=["variable", "value"],
        datatype=["str", "str"],
        interactive=True,
        label="Valeurs utilisateur"
    )

    choosed_features.change(
        fn=update_table,
        inputs=choosed_features,
        outputs=user_table
    )

<<<<<<< Updated upstream

    #gr.Interface(
    #    fn=greet,
    #    inputs=["text", "slider"],
    #    outputs=["text"],
    #    api_name="predict"
    #)
=======
    validate_button = gr.Button("Valider les paramètres")

    types_state = gr.State(value=explicative_features)

    status = gr.Markdown()
    validated_params = gr.JSON(label="Paramètres validés")

    validate_button.click(                  #A terme essayer d'ajouter un gr.render ici
        fn=validate_params,
        inputs=[user_table, types_state],
        outputs=[status, validated_params],
        api_name = "validate_params"
    )

    predict_button = gr.Button("Obtenir les prédictions du modèle")
    prediction = gr.JSON(label="Prédiction")
    database_status = gr.Markdown()   

    predict_button.click(
        fn=infer_from_new_vector,
        inputs=validated_params,
        outputs=[prediction, database_status],
        api_name = "predict"
    )   

    gr.api(
        fn=process_scoring_request,
        api_name="score_client",
    )                        
>>>>>>> Stashed changes

if __name__ == "__main__":
    demo.launch()