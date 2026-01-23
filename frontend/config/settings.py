"""Configuration management for frontend application.

Handles backend URL, API endpoints, timeouts, and default parameters.
"""
import os

# Backend base URL - defaults to localhost for development
# Can be overridden via environment variable or deployment config
BACKEND_BASE_URL: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")

# API endpoint path for chat requests
# Matches the FastAPI router prefix: /api/v1/chat
API_CHAT_ENDPOINT: str = os.getenv("API_CHAT_ENDPOINT", "/api/v1/chat")

# HTTP request timeout in seconds
# Set to 60s to allow for RAG pipeline processing (retrieval + LLM generation)
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))

