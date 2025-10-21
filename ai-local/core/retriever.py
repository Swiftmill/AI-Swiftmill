"""Vector store management and ingestion helpers."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SUPPORTED_TYPES = {".pdf", ".txt", ".md"}


class KnowledgeBase:
    """Wraps the embedding model and vector store."""

    def __init__(self, embed_model: str | None = None) -> None:
        self.embed_model = embed_model or os.getenv("EMBED_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._embedding = SentenceTransformerEmbeddings(model_name=self.embed_model)
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._vectorstore = Chroma(
            collection_name="local-ai-knowledge",
            embedding_function=self._embedding,
            persist_directory=str(INDEX_DIR),
        )

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------
    def ingest_paths(self, paths: Sequence[Path]) -> int:
        """Ingest provided file paths into the vector store.

        Returns the number of chunks added.
        """

        documents: List[Document] = []
        for path in paths:
            if path.suffix.lower() not in SUPPORTED_TYPES:
                continue
            documents.extend(self._load_documents(path))

        if not documents:
            return 0

        self._vectorstore.add_documents(documents)
        self._vectorstore.persist()
        return len(documents)

    def _load_documents(self, path: Path) -> Iterable[Document]:
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            pages = loader.load()
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")
            pages = [Document(page_content=text, metadata={"source": str(path)})]

        chunks = self._text_splitter.split_documents(pages)
        for idx, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "source": str(path),
                    "chunk": idx,
                    "title": path.name,
                }
            )
            yield Document(page_content=chunk.page_content, metadata=metadata)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        if self._vectorstore._collection.count() == 0:
            return []
        return self._vectorstore.similarity_search_with_score(query, k=k)

    def ensure_atomic_copy(self, src: Path, dest: Path) -> None:
        """Copy file contents atomically to the destination path."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as input_fp, tempfile.NamedTemporaryFile(
            "wb", delete=False, dir=str(dest.parent)
        ) as tmp:
            tmp.write(input_fp.read())
            temp_name = Path(tmp.name)
        os.replace(temp_name, dest)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def normalise_filename(filename: str) -> str:
        return filename.replace(" ", "_")


__all__ = ["KnowledgeBase", "INDEX_DIR", "KNOWLEDGE_DIR", "SUPPORTED_TYPES"]
