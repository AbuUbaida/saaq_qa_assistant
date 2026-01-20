"""Answer display component.

Renders formatted answer text, loading states, and metadata (latency, retrieved count).
"""

from typing import Optional

import streamlit as st

from frontend.utils.formatting import safe_markdown


def render_answer(answer: Optional[str]) -> None:
    """Display the answer text with markdown formatting.
    
    Renders the answer prominently using Streamlit's markdown renderer.
    Handles empty or missing answers gracefully by showing nothing.
    
    Args:
        answer: The answer text to display, or None if no answer available.
    """
    # Handle empty or missing answers - don't render anything
    # This prevents showing empty containers or placeholder text
    if not answer or not answer.strip():
        return
    
    # Format answer for safe markdown rendering
    # The safe_markdown function escapes problematic characters while
    # preserving readability and basic formatting
    formatted_answer = safe_markdown(answer)
    
    # Display answer prominently using markdown
    # Using st.markdown allows the answer to include citations and formatting
    # from the RAG pipeline (e.g., [1], [2] citation markers)
    st.markdown(formatted_answer)
