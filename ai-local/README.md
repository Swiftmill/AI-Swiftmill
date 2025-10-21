# AI Local Assistant

Assistant conversationnel local avec FastAPI, Ollama et LangChain. Il répond à vos questions, apprend de vos documents et complète ses réponses grâce à la recherche web.

## Fonctionnalités

- **Chat RAG** (`POST /chat`) : réponse enrichie par vos documents et, si besoin, recherche web.
- **Ingestion de documents** (`POST /ingest`) : ajoutez des fichiers `.pdf`, `.md` ou `.txt` dans `./knowledge` et mettez à jour l'index vectoriel.
- **Mémoire utilisateur** : chaque utilisateur possède un fichier `./data/memory/<user>.json` qui retient les notes demandées (« rappelle-toi… »).
- **Outils intégrés** : recherche DuckDuckGo, résumé de pages web, lecture de documents locaux, prise de notes persistantes.
- **Interface web minimaliste** (mode sombre) : discussion, téléversement drag & drop, historique localStorage, bascule « autoriser la recherche web ».

## Arborescence

```
ai-local/
├── app.py
├── core/
│   ├── llm.py
│   ├── rag.py
│   ├── retriever.py
│   └── tools.py
├── data/
│   ├── index/
│   └── memory/
├── knowledge/
├── ui/
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── .env.example
├── requirements.txt
└── README.md
```

## Prérequis

- Windows 10/11 avec [Python 3.11+](https://www.python.org/downloads/)
- [Ollama](https://ollama.com/) installé et démarré (`ollama --version`).
- Modèle local (ex. `ollama pull llama3`).
- VS Code recommandé pour l'édition et le debug.

## Installation

```powershell
# Cloner ou télécharger le dossier ai-local/
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Copier la configuration par défaut
copy .env.example .env  # adaptez sous Powershell ou utilisez cp sur Git Bash
```

Modifiez `.env` si nécessaire :

- `LLM_MODEL` : nom du modèle Ollama (défaut `llama3`).
- `EMBED_MODEL` : encodeur de phrases (`all-MiniLM-L6-v2`).
- `OLLAMA_HOST` : URL de l'API Ollama (par défaut `http://localhost:11434`).
- `OPENAI_API_KEY` (optionnel) : active un fallback OpenAI si installé.
  - installez `pip install langchain-openai` si vous souhaitez utiliser cette option.

## Démarrage

```powershell
# Depuis le dossier ai-local/
python app.py
# ou
uvicorn app:app --reload
```

Le serveur écoute sur `http://localhost:8000`.

### Interface web

1. Ouvrez `ui/index.html` dans votre navigateur (double clic ou `Open with Live Server` dans VS Code).
2. Discutez avec l'IA, ajoutez vos documents, activez/désactivez la recherche web.
3. L'historique est conservé dans `localStorage`; utilisez un nouvel onglet ou effacez manuellement pour repartir de zéro.

## Ingestion de documents

- Drag & drop sur la zone prévue ou utilisez le bouton « Ajouter des documents ».
- Formats acceptés : `PDF`, `Markdown`, `Texte brut`.
- Limite : 50 Mo par fichier.
- Les fichiers sont copiés dans `./knowledge` et indexés avec Chroma (`./data/index`).

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /chat`

Requête :

```json
{
  "message": "Quel est le dernier rapport ?",
  "user": "user-123",          // optionnel
  "allow_web": true              // autorise la recherche web si nécessaire
}
```

Réponse :

```json
{
  "answer": "...",
  "sources": [
    {"title": "rapport.pdf", "url": "file:///..."},
    {"title": "Article", "url": "https://..."}
  ]
}
```

### `POST /ingest`

Form-data `files[]` (multipart). Retour :

```json
{"status": "ok", "files": ["notes.pdf"], "chunks": 42}
```

## Conseils d'utilisation

- Dites « rappelle-toi que … » pour sauvegarder une note dans votre mémoire personnelle.
- Les résumés web utilisent DuckDuckGo (pas besoin de clé) et un timeout de 10 s.
- Les sources sont toujours listées en fin de réponse (titre + URL/fichier).

## Dépannage

- **Ollama non disponible** : assurez-vous que `ollama serve` est lancé et que `OLLAMA_HOST` pointe vers le bon port.
- **Pas de documents dans les réponses** : vérifiez que `./data/index` est créé après ingestion (`chunks > 0`).
- **Erreur d'import SentenceTransformer** : installez `torch` adapté à votre GPU/CPU si nécessaire.

## Licence

Projet libre à utiliser et adapter pour un usage personnel.
