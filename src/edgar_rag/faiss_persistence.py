import os
import json
import faiss
import numpy as np

def save_faiss_and_metadata(faiss_store, metadata, faiss_path, metadata_path):
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
    # Use the underlying FAISS index, not the wrapper class
    faiss.write_index(faiss_store.index, faiss_path)  # <-- changed from faiss_store.save()
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_faiss_and_metadata(faiss_class, faiss_path, metadata_path):
    if not (os.path.exists(faiss_path) and os.path.exists(metadata_path)):
        return None, None
    # Load the raw FAISS index from file
    index = faiss.read_index(faiss_path)
    # Now instantiate your FaissVectorStore with the loaded index
    faiss_store = faiss_class(index=index)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return faiss_store, metadata

def maybe_load_embeddings(faiss_path, metadata_path, embeddings_path):
    import faiss

    # Check for all files
    if not (os.path.exists(faiss_path) and os.path.exists(metadata_path) and os.path.exists(embeddings_path)):
        return None, None, None

    # Load FAISS
    index = faiss.read_index(faiss_path)
    # Load metadata (JSON)
    with open(metadata_path, 'r', encoding="utf-8") as f:
        metadata = json.load(f)
    # Load embeddings (numpy)
    embeddings = np.load(embeddings_path)
    return index, metadata, embeddings
