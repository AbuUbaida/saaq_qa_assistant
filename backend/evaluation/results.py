"""Evaluation results and CSV export (small + readable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..core.logging import get_logger
from ..core.utils import get_repo_root

logger = get_logger(__name__)


@dataclass(slots=True)
class EvaluationResults:
    experiment_name: str
    timestamp: datetime
    evaluation_data: list[dict[str, Any]]
    scores: dict[str, Any]
    metrics_used: list[str]
    total_test_cases: int
    elapsed_time: float
    pipeline_config: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        detailed_scores = self.scores.get("detailed_scores", []) or []

        rows: list[dict[str, Any]] = []
        for i, item in enumerate(self.evaluation_data):
            row: dict[str, Any] = {
                "test_case_index": i + 1,
                "question": item.get("question", ""),
                "ground_truth_answer": item.get("ground_truth_answer", ""),
                "model_answer": item.get("model_answer", ""),
                "retrieved_contexts_count": len(item.get("retrieved_contexts", []) or []),
                "retrieved_contexts": "; ".join((item.get("retrieved_contexts") or [])[:3]),
                "ground_truth_contexts_count": len(item.get("ground_truth_contexts", []) or []),
                "ground_truth_contexts": "; ".join(item.get("ground_truth_contexts") or []),
                "latency": item.get("latency", 0.0),
                "error": item.get("error", ""),
            }

            if i < len(detailed_scores) and isinstance(detailed_scores[i], dict):
                for metric_name, score in detailed_scores[i].items():
                    row[f"score_{metric_name}"] = score

            rows.append(row)

        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp.isoformat(),
            "total_test_cases": self.total_test_cases,
            "elapsed_time_seconds": self.elapsed_time,
            "average_scores": self.scores.get("average_scores", {}) or {},
            "metrics_used": self.metrics_used,
            "pipeline_config": self.pipeline_config,
        }


def save_results_to_csv(
    results: EvaluationResults,
    output_dir: Optional[Path | str] = None,
    filename: Optional[str] = None,
) -> Path:
    if output_dir is None:
        repo_root = get_repo_root()
        output_dir = repo_root / "data" / "evaluation" / "results"
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp_str = results.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{results.experiment_name}_{timestamp_str}.csv"
    
    output_path = output_dir / filename
    
    df = results.to_dataframe()
    summary = results.summary()

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            f.write(f"# Evaluation Results: {results.experiment_name}\n")
            f.write(f"# Timestamp: {results.timestamp.isoformat()}\n")
            f.write(f"# Total Test Cases: {results.total_test_cases}\n")
            f.write(f"# Elapsed Time: {results.elapsed_time:.2f} seconds\n")
            f.write(f"# Metrics Used: {', '.join(results.metrics_used)}\n")

            avg_scores = summary.get("average_scores", {}) or {}
            if avg_scores:
                f.write("# Average Scores:\n")
                for metric, score in avg_scores.items():
                    f.write(f"#   {metric}: {score:.4f}\n")

            f.write(f"# Pipeline Config: {results.pipeline_config}\n")
            f.write("#\n")

            df.to_csv(f, index=False)

        logger.info("Saved evaluation results to %s", output_path)
        return output_path
    except Exception as exc:
        logger.error("Error saving results to CSV: %s", exc, exc_info=True)
        raise IOError(f"Failed to save results to {output_path}: {exc}") from exc


def get_default_results_dir() -> Path:
    repo_root = get_repo_root()
    return repo_root / "data" / "evaluation" / "results"

