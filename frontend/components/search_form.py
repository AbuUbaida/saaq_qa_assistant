"""Search input form component.

Provides query text input, submit button, and form validation.

This component uses st.form to create a search engine-style interface where
the app only processes queries when explicitly submitted, not on every keystroke.
"""

from typing import Optional

import streamlit as st


def render_search_form() -> Optional[str]:
    """Render a search form and return the submitted query.
    
    Creates a form with a text input and submit button. The form prevents
    automatic reruns on input changes, ensuring the app only processes
    queries when the user explicitly clicks "Search".
    
    Returns:
        The submitted query string if form was submitted, None otherwise.
        Empty strings are normalized to None.
    """
    # Using st.form prevents automatic reruns on every keystroke
    # This is critical for a search engine UI where we want:
    # - One query → one authoritative answer (not continuous updates)
    # - Better performance (no API calls until submission)
    # - Clear user intent (explicit search action)
    # Without st.form, Streamlit would rerun on every character typed,
    # causing unnecessary API calls and poor UX
    with st.form(key="search_form", clear_on_submit=False):
        # Text input for the user's question
        # Using a text_input inside a form prevents reruns until submission
        query = st.text_input(
            label="Search",
            placeholder="Ask a question about SAAQ regulations, procedures, or information...",
            label_visibility="collapsed",  # Hide label for cleaner search engine aesthetic
            key="query_input",
        )
        
        # Submit button - triggers form submission and app rerun
        # Labeled "Search" to match search engine conventions
        submitted = st.form_submit_button(
            label="Search",
            type="primary",  # Primary button styling for main action
            use_container_width=False,
        )
    
    # Return the query only if form was submitted and query is not empty
    # Normalize empty strings to None for easier checking in calling code
    if submitted and query and query.strip():
        return query.strip()
    
    return None

