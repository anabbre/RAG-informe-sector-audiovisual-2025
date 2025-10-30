from pathlib import Path
import json
import uuid
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from informe_sector_audiovisual_2025.embeddings import embed


COLLECTION = "audiovisual_2025"


def load_jsonl(path: Path) -> List[Dict]:
    """Carga un JSONL línea a línea (cada línea = chunk)."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def det_uuid(text: str) -> str:
    """Genera UUID determinista basado en el texto (hash reproducible)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text.strip()))


def main() -> None:
    """
    Inserta los chunks procesados en la colección de Qdrant.
    Pasos:
      - Carga chunks.jsonl
      - Calcula embeddings con Sentence-Transformer
      - Asigna un id hash por chunk
      - Inserta cada punto con su payload en la colección
    """
    src = Path("data/processed/chunks.jsonl")
    if not src.exists():
        raise SystemExit("Falta data/processed/chunks.jsonl (ejecuta chunking).")

    docs = load_jsonl(src)
    texts = [d["text"] for d in docs]
    vectors = embed(texts)

    client = QdrantClient(host="localhost", port=6333)
    points: List[PointStruct] = []

    # Asociar cada embedding con su payload original
    for d, vec in zip(docs, vectors, strict=True):
        hid = det_uuid(d["text"])
        payload = {k: v for k, v in d.items() if k != "id"}
        payload["hash"] = hid  

        points.append(PointStruct(
            id=hid,
            vector=vec,
            payload=payload,
))

    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    print(f"Upsert OK: {len(points)}")


if __name__ == "__main__":
    main()
