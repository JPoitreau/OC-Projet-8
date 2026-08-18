from datetime import datetime
from pathlib import Path
from pickle import load
from time import perf_counter
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from pydantic import TypeAdapter, ValidationError
from sklearn.pipeline import Pipeline

from src.database.database import save_error_log, save_prediction_log

#Commande de lancement du script: python -m src.api.scoring_api
#En local, enregistrement sur csv plutôt que sur PostgreSQL: $env:APP_ENV="remote"

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

def update_table(features: list[str] | str | None):
    """
    Get list of features name and convert it as an empty dataframe.

    Args:
    features: list of features selected by the user
    """
    if not features:
        return pd.DataFrame(
            columns = ["feature", "value"]
        )
    
    if isinstance(features, str):
        # If only one feature is selected (str type), we have to convert it
        # into a list before the dataframe creation
        features = [features]

    return pd.DataFrame({
        "feature" : features,
        "value" : [np.nan]*len(features)
    })

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
        return "No feature selected.", None

    for _, row in df.iterrows():
        feature = row["feature"]
        value = row["value"]

        try:
            adapter = TypeAdapter(eval(types[str(feature)]))
        except KeyError:
            return f"Unknown feature, not suported by pydantic validation : {feature}", None
            

        try:
            parsed[feature] = adapter.validate_python(value)
        except ValidationError as e:
            errors.append(
                f"- `{feature}` : {e.errors()[0]['msg']} — Recieved feature : `{value}`"
            )

    if errors:
        return "Validation error :\n\n" + "\n".join(f"- {e}" for e in errors), None

    return "✅ Features validated.", parsed

def infer_from_new_vector(
        params: dict[str, Any], 
        model: Pipeline | Path | str,
        start_time: float | None = None,
        event_time : datetime | None = None,
        persist: bool = True,
        onnx: bool = False):
    """
    Create new vector from given parameters. Missing parameters are filled
    with nan values and imputed in the model pipeline. Save the prediction
    and the parameters into the database. Return the error message if their
    is a problème

    Args:
    params: dictionnary of features and keys with new values.
    start_time: float used to measure processing time when the function is
    called from process_scoring_request
    """
    if not start_time:
        start_time = perf_counter()

    try:
        new_vector = (pd.DataFrame([params], index=[0])
                    .reindex(columns=data.columns, fill_value=np.nan))

        if not onnx:
            if isinstance(model, str):
                print("inside_condition")
                model = scoring_model 
                prediction = model.predict(new_vector)
            else:
                prediction = model.predict(new_vector)
        else:
            
            import onnxruntime as rt

            session = rt.InferenceSession(model)
            input_name = session.get_inputs()[0].name
            inputs = new_vector.to_numpy(dtype=np.float32)
            prediction = session.run(None, {input_name: inputs})
        
        execution_time_ms = (perf_counter() - start_time) * 1000

        prediction_output = np.asarray(prediction[0] if onnx else prediction)
        predicted_class = int(prediction_output.ravel()[0])

        if persist:
            request_id = save_prediction_log(
                requested_params=params,
                predicted_class=predicted_class,
                execution_time_ms=execution_time_ms,
                event_time = event_time
            )

            message = (
                f"✅ Prediction registered. "
                f"Request ID : '{request_id}'."
            )
        else:
            message = "✅ Prediction not registered."

        return prediction_output.tolist(), message

    except (ValueError, TypeError, KeyError, IndexError) as error:
        execution_time_ms = (perf_counter() - start_time) * 1000

        if persist:
            request_id = save_error_log(
                requested_params={},
                error_message=f'{error}',
                execution_time_ms=0.0,
                event_time = event_time
            )

        return(
            None,
            f"An error occured : '{error}'"
        )

def process_scoring_request(
        user_values: dict[str, Any],
        model: Pipeline | Path | str,
        simulated_event_time: str | None = None,
        persist: bool = True,
        onnx: bool = False
    ):
    """
    Valide les valeurs, effectue la prédiction
    et enregistre la requête (PostgreSQL en local, csv en distant). Permet
    de simuler une requête utlisateur complète à partir d'un dictionnaire
    clés:valeurs
    """

    start_time = perf_counter()

    event_time = datetime.fromisoformat(simulated_event_time)

    if not user_values:
        error_message = "No features value was recieved from the user."

        request_id = save_error_log(
            requested_params={},
            error_message=error_message,
            execution_time_ms=0.0,
            event_time=event_time,
        )

        raise ValueError(
            f"{error_message} Request registered with ID : {request_id}."
        )


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
        execution_time_ms = (perf_counter() - start_time) * 1000

        request_id = save_error_log(
            requested_params=user_values,
            error_message=validation_message,
            execution_time_ms=execution_time_ms,
            event_time=event_time,
        )

        raise ValueError(
            f"{validation_message}\n"
            f"Error registered with ID : {request_id}."
        )

    #Si tout est bon on appel la fonction d'inférence qui sauvegarde la
    #prédiction et retourne la prédiction et la confirmation d'enregistrement.

    prediction, database_message = infer_from_new_vector(
                                            parsed_params,
                                            model, 
                                            start_time,
                                            event_time,
                                            persist,
                                            onnx
                                            )

    #Retourne les résultats dans la console pour le débugage
    return {
        "prediction": prediction,
        "database_status": database_message,
        "validated_params": parsed_params,
    }


with gr.Blocks() as demo:  # pragma: no cover

    choosed_features = gr.Dropdown(
        choices = explicative_features_list,
        multiselect = True,
        value = list
    )
    
    user_table = gr.Dataframe(
        headers=["feature", "value"],
        datatype=["str", "str"],
        interactive=True,
        label="Valeurs utilisateur"
    )

    choosed_features.change(
        fn=update_table,
        inputs=choosed_features,
        outputs=user_table
    )

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
    model_state = gr.State(value=scoring_model)


    predict_button.click(
        fn=infer_from_new_vector,
        inputs=[validated_params, model_state],
        outputs=[prediction, database_status],
        api_name = "predict"
    )   

    gr.api(
        fn=process_scoring_request,
        api_name="score_client",
    )                        

if __name__ == "__main__":  # pragma: no cover
    demo.launch()
