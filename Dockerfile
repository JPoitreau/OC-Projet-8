# Image de base : Python 3.11 avec uv déjà installé
FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim

# Ajout des bibliothèques nécessaires pour LightGBM et PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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