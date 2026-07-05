import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text(pdf_path: str) -> str | None:
    """Extract plain text from a PDF.

    Returns None when the PDF can't be parsed or yields no text — callers skip
    the paper and continue the batch (spec: skip, log, continue).
    """
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        logger.exception("Failed to parse %s", pdf_path)
        return None
    text = "\n".join(line for line in text.splitlines() if line.strip())
    if not text.strip():
        logger.warning("No text extracted from %s", pdf_path)
        return None
    return text
