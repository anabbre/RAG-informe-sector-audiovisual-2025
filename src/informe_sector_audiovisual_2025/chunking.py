from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

"""
Módulo de troceo (chunking) de texto.

Divide documentos largos (por párrafos o frases) en fragmentos
más pequeños, conservando cierta coherencia semántica.

Usado por los scripts de ingesta para generar `chunks.jsonl`.
"""

# ─────── Heurísticas para detectar cabeceras y dividir por frases ───────
HDR_RE = re.compile(r"^(\d+\.\d+)*\s*.+|[A-ZÁÉÍÓÚÜÑ0-9 ,.\-:%()]+$")
SENT_RE = re.compile(r"(?<=\.|\!|\?)\s+(?=[A-ZÁÉÍÓÚÜÑ0-9])", re.MULTILINE)

# Configuración por defecto
DEFAULTS: Dict[str, int] = {
    "chunk_size": 1000,  # ~800–1200 chars 
    "overlap": 150,  # solape para no cortar ideas
}

# ─────── Funciones auxiliares ───────
def _read_txt(path: Path) -> str:
    """Lee texto plano UTF-8."""
    return path.read_text(encoding="utf-8")


def _split_paragraphs(text: str) -> List[str]:
    """Divide por párrafos (líneas en blanco) y normaliza espacios."""
    paras = [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if p]


def _is_header(line: str) -> bool:
    """Detecta cabeceras en mayúsculas o numeradas."""
    line = line.strip()
    if len(line) <= 3:
        return False
    # evita tratar como cabecera frases normales en minúscula sin dígitos
    if line.lower() == line and not re.search(r"\d", line):
        return False
    return bool(HDR_RE.match(line))


def _sentences(paragraph: str) -> List[str]:
    """Divide un párrafo en frases, evitando cortar cabeceras."""
    if _is_header(paragraph):
        return [paragraph]
    # corta por frases (heurística), conservando bullets como "•" o "-"
    tmp = SENT_RE.split(paragraph)
    out: List[str] = []
    for s in tmp:
        s = s.strip()
        if not s:
            continue
        if (s.startswith("•") or s.startswith("-")) and out:
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out

# ─────── Empaquetado de frases en chunks ───────
def _pack_sentences(
    sents: List[str],
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    """
    Agrupa frases en bloques de tamaño máximo (~chunk_size),
    manteniendo solape entre fragmentos.
    """
    chunks: List[Dict[str, Any]] = []
    buf: List[str] = []
    buf_len = 0
    cid = 0

    def flush(add_overlap: bool = True) -> None:
        nonlocal buf, buf_len, cid, chunks
        if not buf:
            return
        text = " ".join(buf).strip()
        if text:
            chunks.append({"id": cid, "text": text})
            cid += 1
        if add_overlap and overlap > 0 and chunks:
            tail = chunks[-1]["text"][-overlap:]
            buf = [tail] if tail else []
            buf_len = len(buf[0]) if buf else 0
        else:
            buf = []
            buf_len = 0

    for s in sents:
        if not s:
            continue
        if buf_len + len(s) + 1 <= chunk_size:
            buf.append(s)
            buf_len += len(s) + 1
        else:
            flush(add_overlap=True)
            # una única frase enorme → partir en trozos "duros"
            if len(s) > chunk_size:
                step = chunk_size - overlap
                for i in range(0, len(s), step):
                    piece = s[i : i + chunk_size]
                    chunks.append({"id": cid, "text": piece})
                    cid += 1
                tail = s[-overlap:]
                buf = [tail] if tail else []
                buf_len = len(tail) if tail else 0
            else:
                buf = [s]
                buf_len = len(s)

    flush(add_overlap=False)
    return chunks

# ─────── Programa principal ───────
def main() -> None:
    """Ejecuta el pipeline completo: lee, trocea y guarda en JSONL."""
    src_txt = Path("data/interim/informe_sector_audiovisual_2025.txt")
    out_jsonl = Path("data/processed/chunks.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    text = _read_txt(src_txt)

    paragraphs = _split_paragraphs(text)
    sents: List[str] = []
    for p in paragraphs:
        sents.extend(_sentences(p))

    cfg = DEFAULTS
    chunks = _pack_sentences(sents, cfg["chunk_size"], cfg["overlap"])

    with out_jsonl.open("w", encoding="utf-8") as f:
        for ch in chunks:
            row = {
                "id": ch["id"],
                "text": ch["text"],
                "source": "informe_sector_audiovisual_2025.pdf",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Chunks: {len(chunks)} -> {out_jsonl.resolve()}")


if __name__ == "__main__":
    main()
