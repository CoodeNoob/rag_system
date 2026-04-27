from dotenv import load_dotenv
import requests
import os
from pathlib import Path
from io import StringIO

import pandas as pd

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


def file_data_embedding(file):
    file_name = (file.filename or "").strip()
    file_extension = Path(file_name).suffix.lower()

    print("Start to make embedding ! ")

    raw_bytes = file.read()
    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")

    text_content = _extract_text_from_file(raw_bytes, file_extension)
    chunks = _chunk_text(text_content)
    print(f"Chunks are : {chunks}")

    if not chunks:
        raise ValueError("No readable text content was found in the uploaded file.")

    from rag_app.data_sourcess.vector_storage import add_documents

    print(f"Data uploading to file")

    add_documents(chunks)


def _extract_text_from_file(raw_bytes, file_extension):
    if file_extension in {".csv"}:
        dataframe = pd.read_csv(StringIO(raw_bytes.decode("utf-8")))
        return dataframe.to_csv(index=False)

    if file_extension in {".txt", ".md", ".json", ".py", ".html", ".css", ".js"}:
        return raw_bytes.decode("utf-8")

    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Unsupported file type. Please upload a UTF-8 text, code, markdown, JSON, or CSV file."
        ) from exc


def _chunk_text(text, chunk_size=800):
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return []

    return [
        normalized_text[i:i + chunk_size]
        for i in range(0, len(normalized_text), chunk_size)
    ]

