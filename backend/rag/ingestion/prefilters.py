from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

class DoclingPage(BaseModel):
    page_no: int
    text: str

def get_devanagari_ratio(text_str: str) -> float:
    """
    Computes the ratio of Devanagari characters to total alphanumeric/punctuation characters
    (ignoring whitespace and numbers).
    """
    if not text_str:
        return 0.0
    # Filter out whitespace and numbers to get a more accurate character density
    chars = [c for c in text_str if not c.isspace() and not c.isdigit()]
    if not chars:
        return 0.0
    devanagari_count = sum(1 for c in chars if '\u0900' <= c <= '\u097F')
    return devanagari_count / len(chars)

def is_devanagari_block(text: str, threshold: float = 0.30) -> bool:
    """
    Determines if a text block is predominantly Hindi/Devanagari.
    If the ratio of Devanagari characters exceeds the threshold (default 30%),
    it returns True.
    """
    ratio = get_devanagari_ratio(text)
    return ratio > threshold

def find_english_classifier_window(docling_pages: list[DoclingPage]) -> str:
    """
    Concatenates all pages of the document and scans for the starting line of the
    continuous English section, resolving cases where the English text starts at
    the bottom of a Hindi page (e.g., bilingual Gazette notifications).
    
    Returns a window of text (~2 pages) starting from the English title.
    """
    # 1. Concatenate all pages into lines to preserve text flow
    all_lines = []
    for page in docling_pages:
        if page.text:
            all_lines.extend(page.text.split('\n'))
            
    if not all_lines:
        return ""

    # 2. Find the first line where the English section begins.
    # To avoid false positives (e.g., brief English headings or names in the Hindi section),
    # we verify that the current line and the subsequent block of text are predominantly English.
    start_line_idx = 0
    total_lines = len(all_lines)
    
    for i in range(total_lines):
        line = all_lines[i].strip()
        # Skip empty lines or extremely short lines (e.g. line numbers) for start detection
        if len(line) < 10:
            continue
            
        if get_devanagari_ratio(line) < 0.15:
            # We found a potential English start. Let's lookahead to verify.
            # Lookahead next 15 lines or 800 characters to confirm it's a sustained English block
            lookahead_lines = all_lines[i:min(i + 15, total_lines)]
            lookahead_text = " ".join(lookahead_lines)
            
            if len(lookahead_text.strip()) > 100 and get_devanagari_ratio(lookahead_text) < 0.15:
                # Confirmed: this is the start of the English section!
                start_line_idx = i
                break
    else:
        # Fallback if no English section detected: start from the beginning
        start_line_idx = 0

    # 3. Return a window of ~2 pages (approx 120 lines or 6000 characters) starting from the English title
    end_line_idx = min(start_line_idx + 120, total_lines)
    return "\n".join(all_lines[start_line_idx:end_line_idx])
