from gradio_client import Client

client = Client("http://127.0.0.1:7860")

client.predict(
    {
        "AMT_CREDIT": 6000,
        "DAYS_BIRTH": -140,
    },
    api_name="/score_client",
)
