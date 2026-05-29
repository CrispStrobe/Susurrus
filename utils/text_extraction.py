# utils/text_extraction.py
"""Text extraction from various file formats for TTS and translation input.

Supports: TXT, MD (Markdown), HTML, PDF, EPUB.

Text extraction patterns adapted from CrispTTS (EUPL v1.2).
Original Copyright (c) CrispStrobe contributors.
"""

import logging
import os


def extract_text(filepath):
    """Extract text from a file based on its extension.

    Raises ImportError if the required library for the file format is missing.
    Raises ValueError for unsupported file formats.
    """
    ext = os.path.splitext(filepath)[1].lower()

    extractors = {
        ".txt": extract_text_from_txt,
        ".md": extract_text_from_md,
        ".html": extract_text_from_html,
        ".htm": extract_text_from_html,
        ".pdf": extract_text_from_pdf,
        ".epub": extract_text_from_epub,
    }

    extractor = extractors.get(ext)
    if not extractor:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported: {', '.join(sorted(extractors))}"
        )

    return extractor(filepath)


def extract_text_from_txt(filepath):
    """Extract text from a plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def extract_text_from_md(filepath):
    """Extract text from a Markdown file, stripping formatting."""
    try:
        import markdown
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "markdown and beautifulsoup4 are required for Markdown extraction. "
            "Install with: pip install markdown beautifulsoup4"
        )

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        md_content = f.read()

    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def extract_text_from_html(filepath):
    """Extract text from an HTML file."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "beautifulsoup4 is required for HTML extraction. "
            "Install with: pip install beautifulsoup4"
        )

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def extract_text_from_pdf(filepath):
    """Extract text from a PDF file."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ImportError(
            "pypdfium2 is required for PDF extraction. "
            "Install with: pip install pypdfium2"
        )

    pdf = pdfium.PdfDocument(filepath)
    text_parts = []
    for page in pdf:
        textpage = page.get_textpage()
        text_parts.append(textpage.get_text_range())
        textpage.close()
        page.close()
    pdf.close()

    return "\n".join(text_parts).strip()


def extract_text_from_epub(filepath):
    """Extract text from an EPUB file."""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "ebooklib and beautifulsoup4 are required for EPUB extraction. "
            "Install with: pip install ebooklib beautifulsoup4"
        )

    book = epub.read_epub(filepath, options={"ignore_ncx": True})
    text_parts = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if text:
            text_parts.append(text)

    return "\n\n".join(text_parts).strip()
