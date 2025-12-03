# embed_utils.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
from pathlib import Path

class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        emb = self.model.encode(texts)
        return np.array(emb, dtype="float32")

class FaissIndex:
    def __init__(self, dim, index_path="faiss_index.index", db_path="metadata.sqlite"):
        self.dim = dim
        self.index_path = index_path
        self.db_path = db_path

        self._init_db()

        if Path(self.index_path).exists():
            try:
                idx = faiss.read_index(self.index_path)
                # Ensure it's wrapped in IndexIDMap
                if not isinstance(idx, faiss.IndexIDMap):
                    idx = faiss.IndexIDMap(idx)
                self.index = idx
            except Exception:
                self._create_index()
        else:
            self._create_index()

    def _create_index(self):
        base = faiss.IndexHNSWFlat(self.dim, 32)
        base.hnsw.efConstruction = 200
        base.hnsw.efSearch = 64
        self.index = faiss.IndexIDMap(base)

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            faiss_id INTEGER UNIQUE,
            doc_id TEXT,
            file_path TEXT,
            page INTEGER,
            chunk_index INTEGER,
            text TEXT,
            cleaned INTEGER DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='chunk_id')")
        conn.commit()
        conn.close()

    def add_documents(self, doc_id, file_path, texts, embeddings, pages=None, chunk_offsets=None):
        """
        texts: list[str] (ordered)
        embeddings: numpy array (N, D)
        pages: optional list of page numbers for each chunk
        chunk_offsets: optional list of chunk_index per chunk
        """
        vecs = np.array(embeddings, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        faiss_ids = []
        for i, txt in enumerate(texts):
            page_val = pages[i] if pages is not None else None
            chunk_idx = chunk_offsets[i] if chunk_offsets is not None else i
            cur.execute("INSERT INTO chunks (doc_id, file_path, page, chunk_index, text) VALUES (?, ?, ?, ?, ?)",
                        (doc_id, file_path, page_val, chunk_idx, txt))
            chunk_id = cur.lastrowid
            faiss_ids.append(chunk_id)
            cur.execute("UPDATE chunks SET faiss_id = ? WHERE chunk_id = ?", (chunk_id, chunk_id))
            cur.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (chunk_id, txt))
        conn.commit()
        conn.close()

        ids_np = np.array(faiss_ids, dtype='int64')
        self.index.add_with_ids(vecs, ids_np)
        faiss.write_index(self.index, self.index_path)

    def search(self, query_emb, top_k=5):
        q = np.array(query_emb, dtype="float32")
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        D, I = self.index.search(np.array([q_norm]).astype("float32"), top_k)
        hits = []
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for dist, fid in zip(D[0], I[0]):
            if int(fid) == -1:
                continue
            cur.execute("SELECT chunk_id, doc_id, file_path, page, chunk_index, text FROM chunks WHERE chunk_id = ?", (int(fid),))
            r = cur.fetchone()
            if r:
                hits.append({"chunk_id": r[0], "doc_id": r[1], "file_path": r[2], "page": r[3], "chunk_index": r[4], "text": r[5], "score": float(dist)})
        conn.close()
        return hits

    def search_ids(self, q_emb, top_k=200):
        q = np.array(q_emb, dtype="float32")
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        D, I = self.index.search(np.array([q_norm]).astype("float32"), top_k)
        return [int(i) for i in I[0] if i != -1]
