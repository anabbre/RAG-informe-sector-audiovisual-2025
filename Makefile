.PHONY: venv install extract chunk create reset upsert query

PY=python

venv:
	python3 -m venv .venv

install: venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

extract:
	. .venv/bin/activate && PYTHONPATH=src $(PY) -m informe_sector_audiovisual_2025.ingest_pdf data/raw/informe_sector_audiovisual_2025.pdf

chunk:
	. .venv/bin/activate && PYTHONPATH=src $(PY) -m informe_sector_audiovisual_2025.chunking

create:
	. .venv/bin/activate && PYTHONPATH=src $(PY) scripts/create_qdrant_collection.py

reset:
	. .venv/bin/activate && PYTHONPATH=src $(PY) scripts/reset_collection.py

upsert:
	. .venv/bin/activate && PYTHONPATH=src $(PY) scripts/upsert_chunks.py

query:
	. .venv/bin/activate && PYTHONPATH=src $(PY) scripts/query_points.py
