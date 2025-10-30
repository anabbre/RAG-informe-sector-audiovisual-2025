from fastapi import FastAPI, Query
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from informe_sector_audiovisual_2025.embeddings import embed

"""
API de consulta mínima sobre la colección vectorial 'audiovisual_2025'.

Permite:
- Comprobar el estado de Qdrant (/health)
- Realizar búsquedas semánticas (/query)
- Obtener información general (/)

Usa el modelo de embeddings definido en `embeddings.py`.
"""

COLLECTION = "audiovisual_2025"

app = FastAPI(title="RAG Audiovisual 2025")

@app.get("/")
def root():
    """Ruta principal: devuelve endpoints disponibles."""
    return {"status": "ok", "message": "Endpoints: /health, /query?q=...&top_k=3, /docs"}

@app.get("/health")
def health():
    """
    Comprueba conexión con Qdrant y devuelve el nombre de la colección.
    Útil para verificar que el contenedor y la API están operativos.
    """
    client = QdrantClient(host="localhost", port=6333)
    _ = client.get_collection(COLLECTION)
    return {"qdrant": "green", "collection": COLLECTION}

@app.get("/query")
def query(
    q: str = Query(..., min_length=1, description="Texto a buscar"),
    top_k: int = Query(3, ge=1, le=10, description="Nº de resultados")
) -> List[Dict[str, Any]]:
    """
    Realiza una búsqueda semántica en Qdrant con el texto proporcionado.

    - Convierte la query a vector con Sentence-Transformers.
    - Busca en la colección definida (payload incluido).
    - Devuelve metadatos (score, chunk_id, page, source...).
    """
    client = QdrantClient(host="localhost", port=6333)
    vec = embed([q])[0]

    res = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=Filter(should=[]),  
    ).points

    out = []
    for r in res:
        pld = r.payload or {}
        out.append({
            "score": r.score,
            "chunk_id": pld.get("chunk_id"),
            "page": pld.get("page"),
            "section": pld.get("section", "")[:300],
            "source": pld.get("source"),
        })
    return out
