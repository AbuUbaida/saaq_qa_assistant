"""Configuration management for frontend application.

Handles backend URL, API endpoints, timeouts, and default parameters.
"""

# Backend base URL - defaults to localhost for development
# Can be overridden via environment variable or deployment config
BACKEND_BASE_URL: str = "http://localhost:8000"

# API endpoint path for chat requests
# Matches the FastAPI router prefix: /api/v1/chat
API_CHAT_ENDPOINT: str = "/api/v1/chat"

# HTTP request timeout in seconds
# Set to 60s to allow for RAG pipeline processing (retrieval + LLM generation)
REQUEST_TIMEOUT: int = 60

