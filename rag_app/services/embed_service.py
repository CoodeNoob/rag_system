from dotenv import load_dotenv
import requests
import os

load_dotenv()

AI_URL = os.getenv("SERVER_URL")
MODEL_NAME = "nomic-embed-text:latest"

if AI_URL is None:
    raise ValueError("SERVER_URL is not set in .env file")

def payload_prepare(question):
    return {
        "model": MODEL_NAME,
        "input": question
    }


def get_embedding(question):
    response = requests.post(
        f"{AI_URL}/api/embed",
        json=payload_prepare(question),
        timeout=30,
    )
    response.raise_for_status()

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise ValueError(
            f"Embedding server did not return JSON. "
            f"Status={response.status_code}, Body={response.text}"
        ) from exc

    embeddings = data.get("embeddings", [])
    if not embeddings:
        raise ValueError(f"No embeddings returned. Response={data}")

    vector = embeddings[0]
    return vector



