from pathlib import Path
import json
import re

SRC_JSONL = Path("data/processed/chunks.jsonl")
INTERIM_JSON = Path("data/interim/informe_sector_audiovisual_2025.json")
OUT_JSONL = SRC_JSONL  # sobreescribimos el mismo fichero

def _norm(s: str) -> str:
    """Normaliza texto (minúsculas y espacios) para mejorar coincidencias."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _load_pages():
    """Carga las páginas del JSON intermedio (salida de ingest_pdf)."""
    with INTERIM_JSON.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    pages = []
    for p in doc.get("pages", []):
        pages.append({
            "num": int(p.get("page", 0)),
            "text": p.get("text", ""),
            "norm": _norm(p.get("text", "")),
        })
    return pages

def _guess_page(chunk_text: str, pages: list[dict]) -> int | None:
    """
    Intenta asociar cada chunk con su página original.
    Heurística:
      - Normaliza el chunk
      - Busca el primer fragmento del chunk que aparezca en alguna página
      - Prueba con diferentes tamaños de snippet (200→80 chars)
    """
    norm = _norm(chunk_text)
    for k in (200, 160, 120, 80):
        snippet = norm[:k]
        if len(snippet) < 40:
            break
        for p in pages:
            if snippet and snippet in p["norm"]:
                return p["num"]
    return None

def main():
    """
    Enriquecedor de metadatos:
      - Lee chunks.jsonl (resultado del chunking)
      - Asigna campo `page` basándose en coincidencias con el texto del PDF
      - Escribe de nuevo el JSONL con el nuevo metadato añadido
    """
    if not SRC_JSONL.exists():
        raise SystemExit(f"No existe {SRC_JSONL}. Ejecuta el chunking primero.")
    if not INTERIM_JSON.exists():
        raise SystemExit(f"No existe {INTERIM_JSON}. Ejecuta ingest_pdf.py primero.")

    pages = _load_pages()
    rows = []
    
    # Leer línea a línea los chunks y añadir page/section/source
    with SRC_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "")
            page = _guess_page(text, pages)
            row = {
                **d,
                "page": page if page is not None else 0,
                "section": text,
                "source": "informe_sector_audiovisual_2025.pdf",
            }
            rows.append(row)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Metadatos añadidos (con página): {len(rows)} -> {OUT_JSONL.resolve()}")

if __name__ == "__main__":
    main()
