"""Utility tools that the assistant can call during reasoning."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
MEMORY_DIR = DATA_DIR / "memory"

MAX_WEB_RESULTS = 3
SEARCH_TIMEOUT = 10
USER_AGENT = "ai-local-assistant/1.0 (https://localhost)"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


def web_search(query: str, max_results: int = MAX_WEB_RESULTS) -> List[Dict[str, str]]:
    """Perform a DuckDuckGo search and return lightweight results."""
    _ensure_dirs()
    results: List[Dict[str, str]] = []

    try:
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results):
                if not hit:
                    continue
                results.append(
                    {
                        "title": hit.get("title") or hit.get("heading") or "",
                        "url": hit.get("href") or hit.get("url") or "",
                        "snippet": hit.get("body") or hit.get("snippet") or "",
                    }
                )
    except Exception:
        return []

    filtered = [item for item in results if item.get("url")]
    return filtered[:max_results]


def browse_url(url: str) -> Optional[Dict[str, str]]:
    """Fetch and summarise a web page."""
    _ensure_dirs()
    if not url.startswith("http"):
        return None

    try:
        response = requests.get(
            url,
            timeout=SEARCH_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    # Clean text content
    for script in soup(["script", "style"]):
        script.decompose()

    text = " ".join(chunk.strip() for chunk in soup.get_text(separator=" ").split())
    text = re.sub(r"\s+", " ", text)

    summary = text[:1500]

    return {"title": title, "text": summary, "url": url}


def read_doc(path: str) -> Optional[str]:
    """Read a document from the local knowledge directory."""
    _ensure_dirs()
    safe_root = KNOWLEDGE_DIR.resolve()
    candidate = (safe_root / path).resolve()

    if not str(candidate).startswith(str(safe_root)) or not candidate.exists():
        return None

    try:
        return candidate.read_text(encoding="utf-8")
    except Exception:
        return None


def save_note(key: str, text: str) -> bool:
    """Persist a short note addressed by *key* to disk."""
    _ensure_dirs()
    if not key:
        return False

    path = MEMORY_DIR / f"{key}.json"
    existing = get_note(key) or {"key": key, "notes": []}
    notes: List[Dict[str, str]] = existing.get("notes", [])  # type: ignore[arg-type]
    notes.append({"text": text})
    data = {"key": key, "notes": notes}

    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            temp_name = tmp.name
        os.replace(temp_name, path)
        return True
    except Exception:
        return False


def get_note(key: str) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """Retrieve the persisted note for *key* if it exists."""
    _ensure_dirs()
    path = MEMORY_DIR / f"{key}.json"
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as fp:
            loaded = json.load(fp)
    except Exception:
        return None

    if not isinstance(loaded, dict):
        return None
    loaded.setdefault("notes", [])
    return loaded


__all__ = ["web_search", "browse_url", "read_doc", "save_note", "get_note"]
