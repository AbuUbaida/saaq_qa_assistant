"""Build a prompt from a user question and retrieved context."""

from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "No relevant context found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        md = doc.metadata or {}
        source_type = md.get("source_type", "unknown")
        if source_type == "pdf":
            source = md.get("source_file") or "unknown file"
            page_number = md.get("page_number")
            label = f"[{i}] Source: {source} ({source_type}), Page {page_number}:"
        elif source_type == "html":
            source = md.get("source_url") or "unknown url"
            label = f"[{i}] Source: {source} ({source_type}):"
        else:
            source = md.get("source_file") or md.get("source_url") or "unknown source"
            label = f"[{i}] Source: {source} ({source_type}):"

        parts.append(f"{label}\n{doc.page_content}")

    return "\n\n".join(parts)


def build_prompt(
    *,
    question: str,
    context_documents: list[Document],
    language: Literal["en", "fr"] = "en",
    answer_style: Literal["concise", "detailed", "formal"] = "concise",
    include_citations: bool = True,
) -> list[BaseMessage]:
    language_instruction = "Answer in English" if language == "en" else "Répondez en français"
    style_instruction = {
        "concise": "Provide a brief, direct answer",
        "detailed": "Provide a comprehensive answer with examples and explanations",
        "formal": "Use formal language and complete sentences",
    }.get(answer_style, "Provide a clear answer")

    if include_citations:
        citation_instruction = (
            "Cite sources using [1], [2], etc. at the end of relevant sentences. "
            "Format references at the end of your answer as: [1] Source URL or filename, page X"
        )
    else:
        citation_instruction = "Do not include citations in your answer."

    system_template = f"""You are a helpful assistant answering questions about Quebec SAAQ.

Use the following retrieved passages to answer the question. If the answer cannot be found in the passages, say so.

Instructions:
- {language_instruction}
- {style_instruction}
- {citation_instruction}
"""

    human_template = """
Context:
{context}

Question: {question}

Answer:
"""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", human_template),
        ]
    )

    context = _format_context(context_documents)
    return prompt_template.format_messages(context=context, question=question)
