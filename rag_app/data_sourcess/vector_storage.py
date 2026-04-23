import faiss
import numpy as np
from rag_app.services.embed_service import get_embedding

documents = [
    "AGGA is an IT company that provides software services across Myanmar.",
    "The CEO of AGGA.IO is Sai Kham Yee."
]

vectors = [get_embedding(doc) for doc in documents]

vectors = np.array(vectors).astype("float32")

faiss.normalize_L2(vectors)


dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(vectors)


def search(query, k=3):
    if len(documents) == 0:
        return []

    k = min(k, len(documents))

    query_vector = np.array([get_embedding(query)]).astype("float32")

    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)

    results = [documents[i] for i in indices[0]]

    return results