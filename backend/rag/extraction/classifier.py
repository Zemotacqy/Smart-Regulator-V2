from backend.rag.extraction.schemas import ClassifierOutput
from backend.rag.extraction.llm_client import call_llm_with_validation
from backend.config import CLASSIFIER_MODEL
import structlog

logger = structlog.get_logger()

CLASSIFIER_SYSTEM_PROMPT = """
You are a regulatory document classifier for the International Financial Services Centres Authority (IFSCA).
Your task is to analyze the provided front-matter text of a document and extract key metadata into JSON.

You must extract:
1. title: The official English title of the document (e.g., "International Financial Services Centres Authority (Banking) Regulations, 2020").
2. doc_type: The document category. Standard categories are: "Act", "Regulation", "Circular", "Framework", "Guidelines", "Master Direction", "Notification", "Order", "FAQ".
3. publish_date: The date the document was notified/published, in ISO format (YYYY-MM-DD).
4. is_amendment: A boolean indicating if this document is an amendment (e.g., has "Amendment" in the title, or amends/substitutes clauses of another regulation).
5. amends_document: If is_amendment is true, extract the title of the original/parent document being amended (e.g., "IFSCA (Banking) Regulations, 2020"). If not an amendment, set this to null.

You must strictly output ONLY a JSON object matching the requested schema. No explanation text, no markdown block wrappers.
"""

async def classify_document(text_window: str) -> ClassifierOutput:
    """
    Invokes the classifier SLM to extract document metadata from a text window.
    
    Args:
        text_window: The front-matter English text from the document.
        
    Returns:
        ClassifierOutput containing title, doc_type, publish_date, is_amendment, and amends_document.
    """
    logger.info("classifying_document_started", text_length=len(text_window))
    
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Document text window:\n---\n{text_window}\n---\nExtract metadata:"}
    ]
    
    result = await call_llm_with_validation(
        model=CLASSIFIER_MODEL,
        messages=messages,
        response_schema=ClassifierOutput
    )
    
    logger.info(
        "classifying_document_completed",
        title=result.title,
        doc_type=result.doc_type,
        is_amendment=result.is_amendment,
        amends_document=result.amends_document
    )
    
    return result
