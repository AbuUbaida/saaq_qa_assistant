"""Evaluation dataset helpers (keep it simple).

The evaluation file format is a JSONL where each line looks like:
  {"question": "...", "answer": "...", "contexts": ["..."], ...optional fields...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..core.utils import get_repo_root

logger = get_logger(__name__)


@dataclass(slots=True)
class TestCase:
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "contexts": self.contexts, **(self.metadata or {})}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TestCase":
        if "question" not in data or "answer" not in data:
            raise KeyError("Each test case must include 'question' and 'answer'")
        question = str(data.get("question", "")).strip()
        answer = str(data.get("answer", "")).strip()
        contexts_raw = data.get("contexts") or []
        contexts = [str(x) for x in contexts_raw] if isinstance(contexts_raw, list) else [str(contexts_raw)]

        metadata = {k: v for k, v in data.items() if k not in {"question", "answer", "contexts"}}
        return TestCase(question=question, answer=answer, contexts=contexts, metadata=metadata)


@dataclass(slots=True)
class EvalTestDataset:
    """Just a thin container so call sites stay readable."""

    test_cases: list[TestCase]

    def __len__(self) -> int:  # pragma: no cover (tiny)
        return len(self.test_cases)


def load_test_cases(file_path: Path | str) -> EvalTestDataset:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Test cases file not found: {path}")

    test_cases: list[TestCase] = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError("JSONL line must be an object")
            test_cases.append(TestCase.from_dict(data))
        except Exception as exc:
            logger.warning("Skipping invalid test case (line=%d, error=%s)", line_num, exc)

    logger.info("Loaded %d test case(s) from %s", len(test_cases), path)
    return EvalTestDataset(test_cases=test_cases)


def get_default_test_cases_path() -> Path:
    return get_repo_root() / "data" / "evaluation" / "test_cases.jsonl"


def create_ragas_dataset(test_dataset: EvalTestDataset):
    """Convert test cases to a RAGAS dataset (ground-truth only).

    Note: This is NOT what we evaluate with (evaluation uses retrieved_contexts + model responses),
    but it's handy to inspect the dataset in a standard format.
    """
    import pandas as pd
    from ragas import EvaluationDataset

    df = pd.DataFrame(
        {
            "question": [tc.question for tc in test_dataset.test_cases],
            "answer": [tc.answer for tc in test_dataset.test_cases],
            "contexts": [tc.contexts if tc.contexts else [""] for tc in test_dataset.test_cases],
        }
    )
    return EvaluationDataset.from_pandas(df)

