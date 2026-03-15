import faiss
import pickle
import numpy as np
import os
from src.embedding_model import get_embedding
from src.dataset_loader import load_cases


BASE_DIR = os.path.dirname(__file__)

print("Loading embedding model...")

index_path = os.path.join(BASE_DIR, "legal_data/legal_index.faiss")
text_path = os.path.join(BASE_DIR, "legal_data/legal_texts.pkl")

# ---- Load existing FAISS database ----
if os.path.exists(index_path) and os.path.exists(text_path):

    print("Loading existing FAISS database...")
    index = faiss.read_index(index_path)
    with open(text_path, "rb") as f:
        cases = pickle.load(f)
else:
    print("Creating FAISS database (first run)...")
    cases = load_cases()[:2000]
    texts = [case["text"] for case in cases]

    embeddings = [get_embedding(text) for text in texts]
    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 40
    
    index.add(np.array(embeddings).astype("float32"))
    
    faiss.write_index(index, index_path)
    with open(text_path, "wb") as f:
        pickle.dump(cases, f)

print("Total cases loaded:", len(cases))

def retrieve_relevant_laws(query_text, k=5):
    query_embedding = [get_embedding(query_text)]
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = 50
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), k
    )

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(cases):
            case = cases[idx]
            text = case["text"]        
            similarity = round(distances[0][i] * 100, 2)

            results.append({
                "title": text.split("\n")[0][:120],
                "content": text,
                "case_name": text.split("\n")[0][:120],
                "main_category": "Supreme Court Judgment",
                "summary": text[:400],
                "similarity": similarity,
                "_id": None
            })

    return results