"""Sources list display component.

Renders expandable source cards with metadata (URL, page number, scores, previews).
"""

from typing import List, Optional

import streamlit as st

from frontend.utils.formatting import format_source_label


def render_sources(sources: Optional[List[dict]]) -> None:
    """Display source references in an expandable section.
    
    Renders sources inside st.expander to keep the UI clean while providing
    access to source information. Each source is displayed as a clickable link
    when a URL is available.
    
    Args:
        sources: List of source dictionaries, or None if no sources available.
    """
    # Handle empty source lists gracefully - don't render expander
    # This prevents showing an empty "Sources" section when there are no sources
    if not sources or len(sources) == 0:
        return
    
    # Use expander to keep sources accessible but not cluttering the main view
    # This follows search engine UX patterns where sources are secondary information
    # Users can expand to see details, but the answer remains the primary focus
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            source_type = source.get("source_type", "")
            source_url = source.get("source_url")
            source_file = source.get("source_file")
            page_number = source.get("page_number")
            content_preview = source.get("content_preview", "")
            
            # Format the source label (filename + page, or URL)
            label = format_source_label(source)
            
            # Create clickable link for HTML sources with URLs
            # For PDFs, display as text since we may not have a direct URL
            # Using markdown links provides native browser link behavior
            if source_type == "html" and source_url:
                # HTML sources: make URL clickable
                st.markdown(f"**[{label}]({source_url})**")
            else:
                # PDF sources or sources without URLs: display as text
                # PDFs are typically local files, so we show the filename and page
                st.markdown(f"**{label}**")
            
            # Display content preview if available
            # This gives users context about why the source is relevant
            # Truncated preview keeps the UI clean while providing useful information
            if content_preview:
                # Use a subtle style for preview text to differentiate from the link
                st.caption(content_preview)

