import os
from dotenv import load_dotenv
import structlog

load_dotenv()

logger = structlog.get_logger()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://manish@localhost/smart_regulator_v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Model configuration
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "ifsca-classifier-3b")
BOUNDARY_MODEL = os.getenv("BOUNDARY_MODEL", "ifsca-boundary-3b")
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "ifsca-extractor-3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:v1.5")
EXPANDER_MODEL = os.getenv("EXPANDER_MODEL", "ifsca-expander-3b")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "ifsca-saullm-7b-ft")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EVAL_MODEL = os.getenv("EVAL_MODEL", "mistral-nemo:12b")

# Known document types for validation
KNOWN_DOC_TYPES = {
    "Act", "Regulation", "Circular", "Framework", "Guidelines",
    "Master Direction", "Notification", "Order", "FAQ"
}

def validate_doc_type(doc_type: str) -> str:
    """
    If doc_type is in the known set, use it as-is.
    If it is new but non-empty, log a warning and accept it.
    If it is empty or null, raise an error.
    """
    if not doc_type:
        raise ValueError("Classifier returned empty doc_type")
    
    cleaned_type = doc_type.strip()
    if cleaned_type not in KNOWN_DOC_TYPES:
        logger.warning("new_doc_type_discovered", doc_type=cleaned_type)
        KNOWN_DOC_TYPES.add(cleaned_type)
    return cleaned_type
