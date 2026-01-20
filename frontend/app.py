"""Streamlit entrypoint for the UI.

Run:
  streamlit run frontend/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

# Add repo root to sys.path BEFORE importing frontend modules
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from frontend.components.answer_display import render_answer
from frontend.components.search_form import render_search_form
from frontend.components.sources_display import render_sources
from frontend.services.api_client import APIError, ask_question
from frontend.utils.state import KEY_ERROR, KEY_LAST_QUERY, KEY_RESULT, initialize_state


def main() -> None:
    st.set_page_config(page_title="SAAQ QA Assistant", layout="wide")
    st.title("SAAQ QA Assistant")

    initialize_state()

    query = render_search_form()
    if query:
        # Clear previous error on new submission
        st.session_state[KEY_ERROR] = None

        # Avoid duplicate API calls on Streamlit reruns
        if query != st.session_state[KEY_LAST_QUERY]:
            st.session_state[KEY_LAST_QUERY] = query
            try:
                with st.spinner("Searching…"):
                    st.session_state[KEY_RESULT] = ask_question(question=query)
            except APIError as exc:
                st.session_state[KEY_RESULT] = None
                st.session_state[KEY_ERROR] = str(exc)

    if st.session_state[KEY_ERROR]:
        st.error(st.session_state[KEY_ERROR])
        return

    result = st.session_state[KEY_RESULT] or {}
    answer = result.get("answer")
    sources = result.get("sources") or []

    render_answer(answer)
    render_sources(sources)


if __name__ == "__main__":
    main()
