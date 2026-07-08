import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import TypeAdapter, ValidationError

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "original" / "demonstration_data.csv"
SCHEMA_PATH = BASE_DIR / "data" / "schema" / "typeAdapters.json"

data = pd.read_csv(DATA_PATH, sep = ";")
features_list = list(data.columns)

explicative_features = pd.read_json(SCHEMA_PATH, typ='series')

features_list_shortened = list(explicative_features.index)


def update_table(features: list[str] | str | None):
    """
    Get list of features name and convert it as an empty dataframe.
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


with gr.Blocks() as demo:

    choosed_features = gr.Dropdown(
        choices = features_list_shortened,
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
        api_name="validate_params",
    )


demo.launch()