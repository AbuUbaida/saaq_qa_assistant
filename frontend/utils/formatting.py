"""Text formatting utilities.

Handles citation formatting, source display formatting, and text sanitization.

Pure functions for text processing - no Streamlit dependencies.
Focuses on readability and safe display of user-facing content.
"""

import re


def clean_whitespace(text: str) -> str:
    """Clean excessive whitespace while preserving paragraph structure.
    
    Removes trailing/leading whitespace and collapses multiple spaces/newlines
    into single spaces/newlines. Preserves intentional paragraph breaks.
    
    Args:
        text: Raw text that may contain excessive whitespace.
    
    Returns:
        Text with normalized whitespace, ready for display.
    """
    if not text:
        return ""
    
    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Collapse multiple newlines (2+) into double newline (paragraph break)
    # This preserves intentional paragraph breaks while removing excessive spacing
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Collapse multiple spaces into single space (but preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)
    
    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    
    # Remove leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    return "\n".join(lines)


def safe_markdown(text: str) -> str:
    """Prepare text for safe markdown rendering.
    
    Escapes markdown special characters that could cause rendering issues
    or security concerns. Preserves basic formatting like line breaks.
    
    Args:
        text: Text that may contain markdown or special characters.
    
    Returns:
        Text safe for Streamlit markdown rendering.
    """
    if not text:
        return ""
    
    # Streamlit's markdown renderer is generally safe, but we escape
    # potentially problematic sequences to ensure consistent display
    # This prevents accidental markdown interpretation in user content
    
    # Escape backticks (code blocks) to prevent formatting issues
    text = text.replace("```", "\\`\\`\\`")
    
    # Escape dollar signs to prevent LaTeX math rendering
    text = text.replace("$", "\\$")
    
    # Clean whitespace for consistent display
    text = clean_whitespace(text)
    
    return text


def format_source_label(source: dict) -> str:
    """Format a source reference into a readable label.
    
    Creates a human-readable label for a source document, showing
    the most relevant identifying information (file name, URL, page).
    
    Args:
        source: Source dictionary with source_type, source_file, source_url, page_number.
    
    Returns:
        Formatted label string (e.g., "1_drivers_handbook.pdf, page 45").
    """
    if not source:
        return ""
    
    source_type = source.get("source_type", "")
    source_file = source.get("source_file")
    source_url = source.get("source_url")
    page_number = source.get("page_number")
    
    # Build label based on source type
    if source_type == "pdf" and source_file:
        label = source_file
        if page_number is not None:
            label += f", page {page_number}"
        return label
    
    elif source_type == "html" and source_url:
        return source_url
    
    # Fallback for incomplete source data
    return source_file or source_url or "Unknown source"

