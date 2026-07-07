import gradio as gr
import os
import pandas as pd
import numpy as np

data_path = "../../data/original"
data = pd.read_csv(os.path.join(data_path, 'demonstration_data.csv'))
variables_list = list(data.columns)
mean_data = data.mean()

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


    #gr.Interface(
    #    fn=greet,
    #    inputs=["text", "slider"],
    #    outputs=["text"],
    #    api_name="predict"
    #)

demo.launch()