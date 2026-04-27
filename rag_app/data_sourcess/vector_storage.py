import faiss
import numpy as np
from rag_app.services.embed_service import get_embedding
import os
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "index.faiss"
META_DATA_PATH = BASE_DIR / "data.pkl"


# GLOBAL
index = None
documents = []


def load_or_create_index():
    global index, documents
    if os.path.exists(INDEX_PATH) and os.path.exists(META_DATA_PATH):
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_DATA_PATH, "rb") as f:
            documents = pickle.load(f)
    else:
        documents = []
        index = None

def add_documents(new_docs):
    global index, documents
    if not new_docs:
        return

    #  Embed new docs
    new_vectors = [get_embedding(doc) for doc in new_docs]
    new_vectors = np.array(new_vectors).astype("float32")

    faiss.normalize_L2(new_vectors)

    # Create index if not exists
    if index is None:
        dimension = new_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)

    index.add(new_vectors)

    documents.extend(new_docs)

    save_index()


#Save index + documents
def save_index():
    if index is not None:
        faiss.write_index(index, str(INDEX_PATH))

    with open(META_DATA_PATH, "wb") as f:
        pickle.dump(documents, f)


def search(query, k=3   ):
    if index is None or len(documents) == 0:
        return []

    k = min(k, len(documents))

    query_vector = np.array([get_embedding(query)]).astype("float32")
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, k)

    results = [documents[i] for i in indices[0]]
    return results


load_or_create_index()


docs = [
    "AGGA is an IT company that provides software services across Myanmar.",
    "The CEO of AGGA.IO is Mr. Sai Kham Yee."
]

add_documents(docs) 
    