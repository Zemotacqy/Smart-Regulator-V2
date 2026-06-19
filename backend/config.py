import os
import json
from pathlib import Path
from dotenv import load_dotenv
import structlog

load_dotenv()

logger = structlog.get_logger()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://manish@localhost/smart_regulator_v3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Model configuration
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "ifsca-classifier-3b")
BOUNDARY_MODEL = os.getenv("BOUNDARY_MODEL", "ifsca-boundary-3b")
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "ifsca-extractor-3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "snowflake-arctic-embed2")
EXPANDER_MODEL = os.getenv("EXPANDER_MODEL", "ifsca-expander-3b")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "ifsca-saullm-7b-ft")
EVAL_MODEL = os.getenv("EVAL_MODEL", "mistral-nemo:12b")

# Gemini API configuration
GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Context budget constraints
MAX_CONTEXT_CHARS = 12000
MAX_SINGLE_SECTION_CHARS = 4000

# Known document types for validation — self-expanding, persisted across restarts
# Sidecar JSON lives in the data/ directory alongside training assets.
_DOC_TYPES_SIDECAR = Path(__file__).parent.parent / "data" / "known_doc_types.json"

_BASE_DOC_TYPES = {
    "Act", "Regulation", "Circular", "Framework", "Guidelines",
    "Master Direction", "Notification", "Order", "FAQ"
}

def _load_doc_types() -> set:
    """Loads the known doc types set, seeding from the JSON sidecar if it exists."""
    types = set(_BASE_DOC_TYPES)
    try:
        if _DOC_TYPES_SIDECAR.exists():
            persisted = json.loads(_DOC_TYPES_SIDECAR.read_text(encoding="utf-8"))
            if isinstance(persisted, list):
                types.update(persisted)
    except Exception as exc:
        logger.warning("doc_types_sidecar_load_failed", error=str(exc))
    return types

def _persist_doc_types(types: set) -> None:
    """Writes the current known doc types to the JSON sidecar atomically."""
    try:
        _DOC_TYPES_SIDECAR.parent.mkdir(parents=True, exist_ok=True)
        _DOC_TYPES_SIDECAR.write_text(
            json.dumps(sorted(types), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("doc_types_sidecar_persist_failed", error=str(exc))

# Initialise the in-memory set from sidecar on module load
KNOWN_DOC_TYPES = _load_doc_types()

def validate_doc_type(doc_type: str) -> str:
    """
    If doc_type is in the known set, use it as-is.
    If it is new but non-empty, log a warning, add to the set, and persist to sidecar.
    If it is empty or null, raise an error.
    """
    if not doc_type:
        raise ValueError("Classifier returned empty doc_type")

    cleaned_type = doc_type.strip()
    if cleaned_type not in KNOWN_DOC_TYPES:
        logger.warning("new_doc_type_discovered", doc_type=cleaned_type)
        KNOWN_DOC_TYPES.add(cleaned_type)
        _persist_doc_types(KNOWN_DOC_TYPES)
    return cleaned_type
