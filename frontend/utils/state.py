"""Session state management helpers.

Handles initialization, state management, and clearing/resetting state.

Centralizing session_state keys here prevents typos and ensures consistent
state management across components. This is especially important in Streamlit
where KeyError exceptions can occur if state keys are accessed before initialization.
"""

from typing import Any, Dict, Optional

import streamlit as st

# Session state keys - centralized to prevent typos and ensure consistency
# Using constants makes refactoring easier and provides IDE autocomplete support
KEY_LAST_QUERY = "last_query"
KEY_RESULT = "result"
KEY_ERROR = "error"


def initialize_state() -> None:
    """Initialize all session state variables with default values.
    
    This function should be called early in the Streamlit app lifecycle
    to prevent KeyError exceptions when accessing session_state keys.
    Initializes defaults for:
    - last_query: The most recent user question (None if no query yet)
    - result: The API response containing answer, sources, and metadata (None if no result)
    - error: Error message string if an API call failed (None if no error)
    
    Why centralize here:
    - Prevents KeyError by ensuring all keys exist before access
    - Single source of truth for state structure
    - Easier to add new state variables in the future
    - Makes state dependencies explicit and discoverable
    """
    # Initialize last_query if not present
    # Stores the user's most recent question for display/retry purposes
    if KEY_LAST_QUERY not in st.session_state:
        st.session_state[KEY_LAST_QUERY] = None
    
    # Initialize result if not present
    # Stores the full API response dict: {answer, sources, retrieved_count, latency}
    if KEY_RESULT not in st.session_state:
        st.session_state[KEY_RESULT] = None
    
    # Initialize error if not present
    # Stores error message string from API exceptions for user feedback
    if KEY_ERROR not in st.session_state:
        st.session_state[KEY_ERROR] = None

