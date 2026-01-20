# SAAQ QA Assistant - Streamlit Frontend

## Project Overview

The SAAQ QA Assistant frontend is a search-style Retrieval-Augmented Generation (RAG) interface for querying Quebec SAAQ (Société de l'assurance automobile du Québec) regulations, procedures, and policy information. The application provides authoritative, source-cited answers to user queries through a search engine-style interface.

This is a **search-style interface**, not a conversational chat interface. Each query produces a single authoritative answer with source citations, following a traditional search engine interaction pattern rather than maintaining conversational context.

## Architecture Overview

The frontend follows a modular architecture with clear separation of concerns:

- **UI Components** (`components/`): Pure rendering components that handle display logic without business logic or state mutation. Components include search form, answer display, and sources display.

- **API Service Layer** (`services/`): HTTP client abstraction that handles all backend communication. Provides typed exceptions for different error scenarios (connection, timeout, HTTP errors) and encapsulates request/response handling.

- **State Management** (`utils/state.py`): Centralized session state management using Streamlit's `st.session_state`. Prevents KeyError exceptions and provides a single source of truth for application state.

- **Configuration** (`config/settings.py`): Constants for backend URL, API endpoints, and request timeouts. Keeps configuration separate from business logic.

The frontend communicates with a FastAPI backend via HTTP POST requests to `/api/v1/chat`. The API client handles serialization, error handling, and response parsing, returning structured dictionaries that match the backend's Pydantic response schemas.

## User Interface Design

The search-style UI was chosen to align with government information system conventions and provide clear, authoritative answers without conversational ambiguity. The interface follows a linear flow: query submission → answer display → source citations.

Key Streamlit components used:
- `st.form`: Prevents automatic reruns on input changes, ensuring queries are only processed on explicit submission
- `st.expander`: Collapsible sources section keeps the UI clean while providing access to source details
- `st.spinner`: Visual feedback during API calls
- `st.error`: Prominent error message display

The layout uses Streamlit's wide layout mode for optimal readability of answers and source information.

## Key Features

- **Search-style query interface**
  - Allows users to submit SAAQ-related queries via a single input form.
  - Implemented using `st.form` and `st.text_input` to prevent unnecessary reruns.
  - Ensures clean UX and controlled backend invocation.

- **Controlled backend API invocation**
  - Prevents duplicate API calls by tracking the last submitted query in `st.session_state`.
  - Compares current query with `KEY_LAST_QUERY` before making HTTP requests.
  - Reduces backend load and improves response times by avoiding redundant processing.

- **Structured error handling**
  - Custom exception hierarchy (`APIConnectionError`, `APITimeoutError`, `APIHTTPError`) provides specific error context.
  - Implemented using Python exception handling and the `requests` library's exception types.
  - Error messages are stored in session state and displayed via `st.error()` for user visibility.

- **Answer display with markdown formatting**
  - Renders API response answers with safe markdown formatting.
  - Uses `st.markdown()` for rendering and `safe_markdown()` utility function for escaping problematic characters.
  - Handles empty or missing answers gracefully by returning early without rendering empty containers.

- **Source citation display**
  - Displays source references in a collapsible `st.expander` component.
  - HTML sources are rendered as clickable markdown links using `[label](url)` syntax.
  - PDF sources display filename and page number as text labels.
  - Uses `format_source_label()` utility for consistent source formatting.

- **Loading state management**
  - Displays `st.spinner` during API calls to provide visual feedback.
  - Maintains result persistence in session state across Streamlit reruns.
  - Prevents UI flicker by preserving previous results while new queries process.

- **Session state persistence**
  - Initializes all state keys early via `initialize_state()` to prevent KeyError exceptions.
  - Stores query results, last query, and error state in `st.session_state`.
  - Enables result persistence across widget interactions and page reruns.

## State Management & Performance

The application uses `st.session_state` to maintain application state across Streamlit reruns. State keys are centralized in `utils/state.py` as constants (`KEY_LAST_QUERY`, `KEY_RESULT`, `KEY_ERROR`) to prevent typos and ensure consistency.

Backend calls are controlled through query comparison: the application only invokes the API when a new query is submitted (different from `KEY_LAST_QUERY`). This prevents duplicate API calls that could occur from:
- Streamlit automatic reruns triggered by widget interactions
- Accidental form double-submission
- Page refresh events

Loading states are handled via `st.spinner()` context manager, which displays a loading indicator during API calls. Errors are stored in session state and displayed using `st.error()`, ensuring error messages persist until a new query is submitted.

No explicit caching is implemented at the frontend level; the backend RAG pipeline handles document retrieval and LLM response generation. Frontend state management ensures results are preserved across reruns without requiring re-fetching.

## Error Handling & Reliability

Empty queries are handled at the form component level: `render_search_form()` returns `None` for empty or whitespace-only inputs, preventing API calls with invalid data.

Backend failures are surfaced through a typed exception hierarchy:
- `APIConnectionError`: Backend service unavailable
- `APITimeoutError`: Request exceeded timeout threshold (60 seconds)
- `APIHTTPError`: Backend returned non-200 status code
- `APIError`: Base exception for all API-related errors

All exceptions are caught in the main application flow, error messages are stored in session state, and displayed to users via `st.error()`. This ensures users receive clear feedback when queries cannot be processed.

For regulatory information systems, reliable error handling is critical because:
- Users need to understand when information is unavailable
- Failed queries should not appear as successful empty results
- Error messages must be actionable (e.g., "backend not running" vs. generic failures)

## Running the Application

### Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the FastAPI backend is running and accessible at `http://localhost:8000` (or update `config/settings.py` with the correct backend URL).

3. Run the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

### Configuration

Backend configuration is managed in `config/settings.py`:
- `BACKEND_BASE_URL`: Backend service URL (default: `http://localhost:8000`)
- `API_CHAT_ENDPOINT`: API endpoint path (default: `/api/v1/chat`)
- `REQUEST_TIMEOUT`: HTTP request timeout in seconds (default: 60)

These can be modified directly in the configuration file or overridden via environment variables in deployment environments.

### Docker Deployment

Build and run using the provided Dockerfile:
```bash
docker build -t saaq-qa-frontend .
docker run -p 8501:8501 saaq-qa-frontend
```

## Frontend Technical Notes

### Architecture Decisions

- **Modular component architecture**: UI components are separated from business logic and state management. Components receive data as parameters and render UI without side effects. This enables independent testing and reuse.

- **Service layer abstraction**: API communication is encapsulated in `services/api_client.py`, providing a clean interface that abstracts HTTP details. The service layer handles serialization, error translation, and timeout management, isolating the UI from HTTP implementation details.

- **Centralized state management**: Session state keys are defined as constants in `utils/state.py` rather than hardcoded strings throughout the codebase. This prevents KeyError exceptions, enables IDE autocomplete, and simplifies refactoring.

- **Configuration separation**: Backend URLs, endpoints, and timeouts are stored in `config/settings.py` as constants. This keeps configuration separate from business logic and makes environment-specific overrides straightforward.

### Modularization Strategy

- **Single responsibility principle**: Each module has a focused purpose: components render UI, services handle API calls, utils provide pure functions, config stores constants. This separation improves maintainability and testability.

- **Pure function design**: Utility functions in `utils/formatting.py` are pure functions with no Streamlit dependencies. This enables unit testing without Streamlit context and allows reuse in non-UI contexts.

- **Component composition**: The main `app.py` orchestrates components rather than embedding UI logic. Components are imported and called with data, following a composition pattern that simplifies the main application flow.

### API Integration Patterns

- **Typed exception hierarchy**: Custom exceptions (`APIConnectionError`, `APITimeoutError`, `APIHTTPError`) provide specific error context. The exception hierarchy allows the main application to handle different error types appropriately while maintaining a clean API client interface.

- **Request/response abstraction**: The API client accepts Python dictionaries and returns Python dictionaries, abstracting JSON serialization. This keeps the UI layer unaware of HTTP details while maintaining type safety through structured response dictionaries.

- **Timeout and error handling**: HTTP timeouts are configured at 60 seconds to accommodate RAG pipeline processing (document retrieval + LLM generation). Connection errors, timeouts, and HTTP errors are caught and translated to user-friendly error messages.

### UX Decisions

- **Search-style interface**: Chosen over chat interface to provide authoritative, single-answer responses with source citations. Aligns with government information system conventions and reduces ambiguity in regulatory information delivery.

- **Form-based query submission**: `st.form` prevents automatic reruns on input changes, ensuring queries are only processed on explicit submission. This prevents unnecessary API calls and provides clear user intent.

- **Expander for sources**: Sources are displayed in `st.expander` to keep the main UI focused on answers while providing access to source details. This follows search engine UX patterns where sources are secondary information.

- **Wide layout mode**: Streamlit wide layout provides optimal horizontal space for reading answers and source information, improving readability for longer regulatory text.

### Robustness Considerations

- **State initialization**: All session state keys are initialized early via `initialize_state()` to prevent KeyError exceptions. This ensures safe state access throughout the application lifecycle.

- **Query deduplication**: Last query tracking prevents duplicate API calls for the same query, reducing backend load and improving response times. This handles edge cases like accidental double-submission or Streamlit reruns.

- **Graceful degradation**: Empty queries, missing answers, and empty source lists are handled gracefully without rendering empty UI elements. Components return early when data is unavailable, preventing visual artifacts.

- **Error state management**: Errors are stored in session state and cleared when new queries are submitted. This ensures error messages persist until explicitly replaced, providing clear user feedback.

- **Exception coverage**: All API exception types are caught and handled, including unexpected exceptions. This prevents application crashes and ensures users always receive feedback, even for unexpected errors.

## Future Improvements

- **Query filters**: Add filters for document types (PDF vs. HTML), date ranges, or specific SAAQ regulation categories.

- **Multilingual support**: Enhance language selection (currently English/French) with UI language switching and improved French language model support.

- **Query history**: Persist query history in session state or local storage to enable query reuse and result comparison.

- **Analytics integration**: Add anonymous usage analytics to track common queries and identify knowledge base gaps.

- **Advanced source display**: Enhance source display with relevance scores, content preview expansion, and direct document navigation.
