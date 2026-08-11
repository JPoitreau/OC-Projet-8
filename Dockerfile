# Image de base : Python 3.10 avec uv déjà installé
FROM ghcr.io/astral-sh/uv:python3.10-trixie-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# On n'installe pas les dépendances de développement
ENV UV_NO_DEV=1

# On copie le projet dans l'image
COPY . /app

# On installe les dépendances depuis uv.lock
RUN uv sync --locked

# On utilise l'environnement virtuel créé par uv
ENV PATH="/app/.venv/bin:$PATH"

# Port utilisé par Gradio / Hugging Face Spaces
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Commande lancée au démarrage du conteneur
CMD ["python", "-m", "src.api.scoring_api"]