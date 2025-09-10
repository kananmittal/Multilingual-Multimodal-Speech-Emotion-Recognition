import numpy as np
import os

def load_embeddings(npz_path):
    """
    Load embeddings, labels, and metadata from a .npz file.
    """
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    file_paths = data["file_paths"]
    embedding_type = str(data["embedding_type"])
    embedding_dim = int(data["embedding_dim"])
    num_samples = int(data["num_samples"])

    print(f"[INFO] Loaded {num_samples} samples with {embedding_dim}-dim {embedding_type} embeddings")
    return embeddings, labels, file_paths
