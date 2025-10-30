from pathlib import Path
from typing import Dict, Any
import json
import sys

"""
Extracción de texto desde PDF a ficheros intermedios:
  - data/interim/<nombre>.json  (páginas + texto)
  - data/interim/<nombre>.txt   (texto plano concatenado)
Usa PyMuPDF si está disponible; si no, cae a pdfminer.six. 
"""

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def extract_with_pymupdf(pdf_path: Path) -> Dict[str, Any]:
    """Extrae páginas con PyMuPDF (rápido y robusto)."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page": i + 1, "text": text})
    return {
        "num_pages": len(doc),
        "pages": pages,
        "meta": dict(doc.metadata or {}),
    }

def extract_with_pdfminer(pdf_path: Path) -> Dict[str, Any]:
    """Alternativa basada en pdfminer.six si PyMuPDF no está disponible."""
    from pdfminer.high_level import extract_text
    from pdfminer.pdfparser import PDFSyntaxError

    text = extract_text(str(pdf_path))
    pages = [{"page": i + 1, "text": t} for i, t in enumerate(text.split("\f"))]
    return {
        "num_pages": len(pages),
        "pages": pages,
        "meta": {},
    }

def extract_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Elige el backend de extracción más adecuado disponible en runtime."""
    if fitz is not None:
        return extract_with_pymupdf(pdf_path)
    # if PyMuPDF missing, use pdfminer
    return extract_with_pdfminer(pdf_path)

def save_outputs(pdf_path: Path, result: Dict[str, Any]):
    """
    Guarda:
      - JSON con listado de páginas
      - TXT con todas las páginas concatenadas (para chunking posterior)
    """
    out_base = pdf_path.stem
    out_json = pdf_path.parent.parent / "interim" / f"{out_base}.json"
    out_txt = pdf_path.parent.parent / "interim" / f"{out_base}.txt"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # TXT plano
    joined = "\n\n".join(p["text"].strip() for p in result["pages"])
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(joined)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")

def main():
    """
    Uso:
        python -m informe_sector_audiovisual_2025.ingest_pdf data/raw/<informe>.pdf
    """
    if len(sys.argv) < 2:
        print("Usage: python -m informe_sector_audiovisual_2025.ingest_pdf <path_to_pdf>")
        sys.exit(1)
    pdf_path = Path(sys.argv[1]).resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(2)
    result = extract_pdf(pdf_path)
    save_outputs(pdf_path, result)

if __name__ == "__main__":
    main()
