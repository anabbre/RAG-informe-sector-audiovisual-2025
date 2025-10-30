from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from informe_sector_audiovisual_2025.embeddings import embed

COLLECTION = "audiovisual_2025"


def main():
    """
    Script de prueba rápida para comprobar que la búsqueda semántica funciona.
    - Genera embedding de una frase ejemplo
    - Realiza query en Qdrant (top=3)
    - Muestra el texto más relevante y su score
    """
    q = "tendencias de empleo en el sector audiovisual en 2025"
    client = QdrantClient(host="localhost", port=6333)
    vec = embed([q])[0]

    res = client.query_points(
        collection_name=COLLECTION,
        query=vec,  
        limit=3,
        with_payload=True,
        with_vectors=False,
        query_filter=Filter(should=[]),  # sin filtros adicionales
    ).points

    for i, r in enumerate(res, 1):
        print(f"[{i}] score={r.score:.4f}")
        print(r.payload.get("text", "")[:400], "\n---\n")


if __name__ == "__main__":
    main()
