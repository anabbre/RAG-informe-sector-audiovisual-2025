from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, OptimizersConfigDiff, VectorParams
from informe_sector_audiovisual_2025.embeddings import dim

COLLECTION = "audiovisual_2025"

def main():
    """
    Script de inicialización de colección en Qdrant.

    - Verifica si la colección existe (para evitar recrearla).
    - Si no existe, la crea con parámetros adecuados:
        * tamaño del vector = dim() (embedding model)
        * métrica = coseno
        * optimizadores = configuración mínima por defecto
    - Crea también un índice de texto (`MatchText`) sobre el campo 'text' 
      para permitir filtrados semánticos.
    """
    client = QdrantClient(host="localhost", port=6333)

    # Verificar si ya existe la colección
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"✅ La colección '{COLLECTION}' ya existe.")
    else:
        # Crear nueva colección
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim(), distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(default_segment_number=2),
        )
        print(f"✅ Creada la colección '{COLLECTION}' con dim={dim()}")

    # Crear índice de texto para búsquedas con MatchText
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="text",
            field_schema="text",  # permite búsqueda por texto completo
        )
        print("🔎 Índice de texto creado sobre el campo 'text'")
    except Exception as e:
        print(f"⚠️ No se pudo crear el índice (posiblemente ya exista): {e}")

if __name__ == "__main__":
    main()

