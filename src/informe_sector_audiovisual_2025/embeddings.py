from __future__ import annotations
from typing import List
import os
from sentence_transformers import SentenceTransformer

"""
Capa del modelo de embeddings (encapsulada para que el resto del código no tenga que preocuparse de cachés/instanciación).
"""

DEFAULT_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
_model = None

def get_model() -> SentenceTransformer:
    """Carga perezosa del modelo (singleton en memoria del proceso)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(DEFAULT_MODEL)
    return _model

def embed(texts: List[str]) -> List[List[float]]:
    """Devuelve los embeddings *normalizados* de una lista de textos."""
    return get_model().encode(texts, normalize_embeddings=True).tolist()

def dim() -> int:
    """Dimensionalidad del espacio de embeddings."""
    return int(get_model().get_sentence_embedding_dimension())
