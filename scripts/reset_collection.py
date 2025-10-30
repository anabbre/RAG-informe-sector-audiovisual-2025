from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

COLLECTION = "audiovisual_2025"


def main():
    """
    Reinicia completamente una colección Qdrant:
      - Si existe, la elimina
      - La vuelve a crear con tamaño de vector 384 y métrica coseno
    Ideal para limpiar la base antes de una nueva ingesta.
    """
    client = QdrantClient(host="localhost", port=6333)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"Eliminada colección '{COLLECTION}'")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"Creada colección '{COLLECTION}' (dim=384, cosine)")


if __name__ == "__main__":
    main()
