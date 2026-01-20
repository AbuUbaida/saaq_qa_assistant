"""RAGAS-based evaluation module for SAAQ QA RAG system.

This module provides components for evaluating RAG pipeline performance using
RAGAS metrics, focusing on the RAG Triad:
- Faithfulness: Measures if the answer is grounded in the provided context
- Answer Relevance: Measures how relevant the answer is to the question
- Context Precision: Measures whether all retrieved contexts are relevant to the question
"""

from __future__ import annotations

from .metrics import create_metrics, create_ragas_llm_and_embeddings
from .dataset import (
    EvalTestDataset,
    load_test_cases,
    create_ragas_dataset,
    get_default_test_cases_path,
)
from .experiment import run_evaluation
from .results import EvaluationResults, save_results_to_csv, get_default_results_dir

__all__ = [
    "create_metrics",
    "create_ragas_llm_and_embeddings",
    "EvalTestDataset",
    "load_test_cases",
    "create_ragas_dataset",
    "get_default_test_cases_path",
    "run_evaluation",
    "EvaluationResults",
    "save_results_to_csv",
    "get_default_results_dir",
]

