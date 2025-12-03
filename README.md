# Biblo Search Context (version2)

Professional document search, OCR, embedding and chat system built with FastAPI, FAISS and Sentence-Transformers.

**Repository:** local project for extracting text from PDFs/images/DOCX/TXT using OCR, embedding chunks for semantic search, and providing a chat + search web UI.

**Status:** Work-in-progress — this repo contains an advanced search & chat prototype. See requirements and notes below.

**Key Features**
- **OCR extraction** from PDF/images using `pytesseract`, `easyocr` (Arabic & English focused).
- **Text cleaning & deduplication** tailored for OCR noise and Arabic/English normalization.
- **Semantic embeddings** using `sentence-transformers` and a local FAISS index for fast vector search.
- **Hybrid search** (FTS + semantic) with pagination and snippet highlighting.
- **Simple chat endpoint** backed by a queued GPT/GPT4All worker (`chat_queue.py`).
- **Web UI** served from `index.html` for file upload, searching and chat.

**Repository layout**
- `app.py` — FastAPI app and endpoints (upload, search, chat, file serving).
- `ocr_utils.py` — Extraction & OCR helpers.
- `embed_utils.py` — `LocalEmbedder` and `FaissIndex` wrapper.
- `utils.py` — cleaning, chunking, embedding helpers.
- `chat_queue.py` — background queue and GPT4All HTTP worker.
- `uploads/` — uploaded files (data, not tracked by git).
- `faiss_index.index`, `metadata.sqlite` — generated index and DB (do not commit).

**Prerequisites**
- Python 3.10+ (3.11 recommended).
- System dependencies:
  - `poppler` (for `pdf2image`) — on Windows install Poppler and set `POPPLER_PATH` in `.env` or in `ocr_utils.py`.
  - Tesseract OCR — install and set `TESSERACT_PATH` if needed.

**Quick setup (recommended: use a venv)**

1. Create and activate a virtual environment (Windows `bash.exe`):

```bash
python -m venv .venv
source .venv/Scripts/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` (optional) to override defaults (see Environment variables).

4. Run the app (development):

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://127.0.0.1:8000/` in your browser.

**Environment variables**
Create a `.env` file or export values in your environment. Common variables used by the app:

- `FAISS_INDEX_PATH` — path to FAISS index file (default `faiss_index.index`).
- `SQLITE_DB` — sqlite DB path for metadata (default `metadata.sqlite`).
- `EMBED_MODEL` — embedding model name for `sentence-transformers` (default `all-MiniLM-L6-v2`).
- `OLLAMA_URL`, `OLLAMA_MODEL` — optional LLM endpoint settings.
- `GPT4ALL_URL`, `GPT4ALL_MODEL` — used by `chat_queue.py` if you route chat requests to a local GPT4All server.

**Important files to ignore (they are large or sensitive)**
- `faiss_index.index`, `metadata.sqlite`, `uploads/`, `.env`, virtual env dirs, and `__pycache__` — see `.gitignore`.

**Notes & Gotchas**
- OCR and EasyOCR may require a GPU or substantial CPU; `easyocr` loads `torch` and models.
- On Windows set `POPPLER_PATH` and `TESSERACT_PATH` in `ocr_utils.py` or via `.env` accordingly.
- FAISS may be installed as `faiss-cpu` or `faiss-gpu` depending on your environment.

**Security & Privacy**
- Do NOT commit `uploads/` or any sensitive documents to version control. Keep `faiss_index.index` and `metadata.sqlite` out of git.

**Contributing**
- If you want to extend the project, please open issues or PRs. Keep changes focused and add tests for new behavior.

**How to push this project to GitHub (step-by-step)**

1. Initialize a local git repository (if not already initialized):

```bash
git init
```

2. Create a `.gitignore` (this repo already contains one). Verify it includes `faiss_index.index`, `metadata.sqlite`, and `uploads/`.

3. Stage and commit your files:

```bash
git add .
git commit -m "Initial commit: Biblo Search Context v2"
```

4. Create a remote GitHub repository:
   - Option A (GitHub website): create a new repo and copy the `git remote add` command shown.
   - Option B (GitHub CLI):

```bash
gh repo create my-repo-name --public --source=. --remote=origin --push
```

5. If you created repo via the website, add the remote and push:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

6. Verify on GitHub that files are present. If you need to push large files later, use Git LFS for binaries.

**Windows (Git Bash) notes**
- Commands above work under `bash.exe` (Git Bash); if using PowerShell replace `source .venv/Scripts/activate` with `.\\.venv\\Scripts\\Activate.ps1`.

**Example minimal workflow to push now (copy-paste into Git Bash)**

```bash
cd "d:/my main laptop/Geo Makanii/SGS project/Biblo Search Context/version2"
git init
git add .
git commit -m "Initial commit: add search/chat prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Replace `<your-username>` and `<repo-name>` with your GitHub account and desired repo name.

**Optional: create a release or tag**

```bash
git tag -a v0.1 -m "v0.1 initial prototype"
git push origin v0.1
```

**Support**
If you want, I can:
- create a `README.md` (done), `.gitignore` (done), and `requirements.txt` (done),
- help you create a GitHub repo and push (I can provide step-by-step commands or a script),
- add a minimal Dockerfile or `devcontainer.json` for reproducible development.

---

If you'd like, I can now:
- run a quick check for missing dependencies in `requirements.txt`, or
- create a `Dockerfile` and `Makefile` to simplify running the app.

Tell me which next step you'd like.

