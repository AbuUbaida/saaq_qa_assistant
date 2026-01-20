"""Run RAG evaluation with RAGAS (kept small).

This module intentionally exposes one main function: `run_evaluation(...)`.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional

from .dataset import EvalTestDataset
from .results import EvaluationResults

def run_evaluation(
    test_dataset: EvalTestDataset,
    metrics: Optional[list[Any]] = None,
    experiment_name: Optional[str] = None,
    search_method: Optional[str] = None,
    top_k: Optional[int] = None,
    **pipeline_kwargs: Any,
) -> EvaluationResults:
    """Run evaluation on test cases and return a structured result."""
    from ragas import EvaluationDataset
    from ragas import evaluate

    import pandas as pd

    from ..core.logging import get_logger
    from ..core.config import get_settings
    from ..db.vector_store import WeaviateStore
    from ..services.document_retriever import retrieve_documents
    from ..services.rag_pipeline import answer_question
    from .metrics import create_metrics, create_ragas_llm_and_embeddings

    logger = get_logger(__name__)

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty.")

    settings = get_settings()
    experiment_name = experiment_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    effective_search_method = search_method or "hybrid"
    effective_top_k = top_k or settings.default_top_k

    if metrics is None:
        llm, embeddings = create_ragas_llm_and_embeddings()
        metrics = create_metrics(llm=llm, embeddings=embeddings)

    start_time = time.time()

    evaluation_data: list[dict[str, Any]] = []

    store = WeaviateStore(
        use_local=settings.weaviate_use_local,
        url=settings.weaviate_cloud_url,
        api_key=settings.weaviate_api_key,
        collection_name=settings.weaviate_collection,
    )
    try:
        for i, test_case in enumerate(test_dataset.test_cases, start=1):
            try:
                response = answer_question(
                    question=test_case.question,
                    search_method=effective_search_method,  # type: ignore[arg-type]
                    top_k=effective_top_k,
                    **pipeline_kwargs,
                )

                retrieved_docs = retrieve_documents(
                    weaviate_client=store.client,
                    query=test_case.question,
                    search_method=effective_search_method,  # type: ignore[arg-type]
                    top_k=effective_top_k,
                    collection_name=settings.weaviate_collection,
                    embedding_model=settings.embedding_model,
                    api_key=settings.hf_api_key,
                )
                full_contexts = [doc.page_content for doc in retrieved_docs]

                evaluation_data.append(
                    {
                        "question": test_case.question,
                        "ground_truth_answer": test_case.answer,
                        "model_answer": response.get("answer", ""),
                        "retrieved_contexts": full_contexts,
                        "ground_truth_contexts": test_case.contexts,
                        "latency": float(response.get("latency", 0.0) or 0.0),
                    }
                )
            except Exception as exc:
                logger.warning("Error in test case %d: %s", i, exc)
                evaluation_data.append(
                    {
                        "question": test_case.question,
                        "ground_truth_answer": test_case.answer,
                        "model_answer": "",
                        "retrieved_contexts": [],
                        "ground_truth_contexts": test_case.contexts,
                        "latency": 0.0,
                        "error": str(exc),
                    }
                )
    finally:
        store.close()

    # RAGAS expects these columns for the common metrics.
    df = pd.DataFrame(
        {
            "user_input": [item["question"] for item in evaluation_data],
            "response": [item["model_answer"] for item in evaluation_data],
            "retrieved_contexts": [
                item["retrieved_contexts"] if item["retrieved_contexts"] else [""]
                for item in evaluation_data
            ],
            "reference": [item.get("ground_truth_answer", "") for item in evaluation_data],
        }
    )
    ragas_dataset = EvaluationDataset.from_pandas(df)

    result = evaluate(dataset=ragas_dataset, metrics=metrics)
    scores_df = result.to_pandas()

    avg_scores: dict[str, float] = {}
    for metric in metrics:
        name = getattr(metric, "name", None) or str(metric)
        if name in scores_df.columns:
            avg_scores[name] = float(scores_df[name].mean())

    elapsed_time = time.time() - start_time
    return EvaluationResults(
        experiment_name=experiment_name,
        timestamp=datetime.now(),
        evaluation_data=evaluation_data,
        scores={
            "detailed_scores": scores_df.to_dict("records"),
            "average_scores": avg_scores,
            "full_results": result,
        },
        metrics_used=[getattr(m, "name", None) or str(m) for m in metrics],
        total_test_cases=len(test_dataset),
        elapsed_time=elapsed_time,
        pipeline_config={
            "search_method": effective_search_method,
            "top_k": effective_top_k,
            **pipeline_kwargs,
        },
    )

