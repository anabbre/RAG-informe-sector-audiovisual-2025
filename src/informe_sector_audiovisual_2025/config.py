from pathlib import Path

# Rutas de proyecto (útiles si en el futuro añadimos más scripts/notebooks).
# Se resuelven relativas al paquete `informe_sector_audiovisual_2025`.
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
