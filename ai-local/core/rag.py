"""Retrieval-augmented generation pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

from .llm import get_chat_model
from .retriever import KnowledgeBase
from .tools import browse_url, get_note, save_note, web_search


@dataclass
class Answer:
    text: str
    sources: List[Dict[str, str]]


class RAGPipeline:
    """Coordinates retrieval, reasoning, and memory."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self.chat_model = get_chat_model()
        self.score_threshold = 0.4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        user_id: Optional[str] = None,
        allow_web: bool = True,
    ) -> Answer:
        """Generate an answer for *question*."""

        just_memorised = False
        if user_id:
            just_memorised = self._maybe_update_memory(user_id, question)
        user_memory = self._load_user_memory(user_id)
        docs_with_scores = self.knowledge_base.similarity_search_with_score(question, k=4)
        rag_docs = [doc for doc, _ in docs_with_scores]
        best_score = min((score for _, score in docs_with_scores), default=None)
        low_confidence = best_score is None or best_score > self.score_threshold

        context = self._build_context(rag_docs, user_memory)
        sources: List[Dict[str, str]] = self._collect_sources_from_docs(rag_docs)

        web_context = ""
        web_sources: List[Dict[str, str]] = []
        if allow_web and (low_confidence or self._requires_web(question)):
            web_context, web_sources = self._augment_with_web(question)
            sources.extend(web_sources)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Tu es un assistant IA français. Réponds de manière structurée et concise. "
                        "Utilise le contexte fourni (connaissances locales, mémoire utilisateur, web). "
                        "Si une information est absente, indique-le."
                    ),
                ),
                (
                    "human",
                    (
                        "Question: {question}\n\n"
                        "Mémoire mise à jour: {memorised}\n"
                        "Mémoire utilisateur: {memory}\n\n"
                        "Contexte local: {context}\n\n"
                        "Recherche web: {web}\n\n"
                        "Fournis la meilleure réponse possible."
                    ),
                ),
            ]
        )

        chain = prompt | self.chat_model
        llm_response = chain.invoke(
            {
                "question": question,
                "memorised": "oui" if just_memorised else "non",
                "memory": user_memory or "(aucune)",
                "context": context or "(aucun)",
                "web": web_context or "(non utilisée)",
            }
        )

        text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

        final_answer = self._append_sources(text, sources)
        return Answer(text=final_answer, sources=sources)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def _load_user_memory(self, user_id: Optional[str]) -> str:
        if not user_id:
            return ""
        note = get_note(user_id)
        if not note:
            return ""
        notes = [item.get("text", "") for item in note.get("notes", [])]
        return "\n".join(filter(None, notes))

    def _maybe_update_memory(self, user_id: str, message: str) -> bool:
        lowered = message.lower()
        if "rappelle-toi" not in lowered and "souviens-toi" not in lowered:
            return False
        match = re.search(
            r"(?:rappelle-toi|souviens-toi)\s*(?:que)?\s*(.*)",
            message,
            flags=re.IGNORECASE,
        )
        if not match:
            return False
        remembered = match.group(1).strip()
        if remembered:
            save_note(user_id, remembered)
            return True
        return False

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _build_context(self, docs: Sequence[Document], memory: str) -> str:
        sections: List[str] = []
        if memory:
            sections.append(f"Notes personnelles:\n{memory}")
        for doc in docs:
            title = doc.metadata.get("title") or Path(doc.metadata.get("source", "")).name
            snippet = doc.page_content.strip()
            sections.append(f"{title}:\n{snippet}")
        return "\n\n".join(sections)

    def _collect_sources_from_docs(self, docs: Sequence[Document]) -> List[Dict[str, str]]:
        collected: Dict[str, Dict[str, str]] = {}
        for doc in docs:
            source_path = doc.metadata.get("source")
            if not source_path:
                continue
            key = str(Path(source_path).resolve())
            if key in collected:
                continue
            collected[key] = {
                "title": doc.metadata.get("title") or Path(source_path).name,
                "url": f"file://{key}",
            }
        return list(collected.values())

    def _requires_web(self, question: str) -> bool:
        lowered = question.lower()
        triggers = ["actualité", "news", "web", "internet", "en ligne"]
        return any(trigger in lowered for trigger in triggers)

    def _augment_with_web(self, question: str) -> Tuple[str, List[Dict[str, str]]]:
        search_hits = web_search(question, max_results=3)
        sections: List[str] = []
        sources: List[Dict[str, str]] = []
        for hit in search_hits:
            summary = browse_url(hit["url"]) if hit.get("url") else None
            if not summary:
                continue
            sections.append(f"{summary['title']}: {summary['text']}")
            sources.append({"title": summary["title"], "url": summary["url"]})
        return "\n".join(sections), sources

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _append_sources(self, answer: str, sources: List[Dict[str, str]]) -> str:
        if not sources:
            return answer
        lines = ["Sources:"]
        seen = set()
        for source in sources:
            key = (source.get("title"), source.get("url"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {source.get('title', 'Source')} ({source.get('url', '')})")
        return answer.strip() + "\n\n" + "\n".join(lines)


__all__ = ["RAGPipeline", "Answer"]
