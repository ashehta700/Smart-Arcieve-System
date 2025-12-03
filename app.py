# app.py - Enhanced version with better search, pagination, and chat
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Query as QueryParam
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import sqlite3
import numpy as np
import requests
import urllib.parse
import json
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from datetime import datetime

from ocr_utils import extract_text
from embed_utils import LocalEmbedder, FaissIndex
from utils import clean_text, embed_texts, chunk_text_intelligently, clean_ocr_noise ,is_meaningful_text
from chat_queue import submit_prompt, start_gpt4all_worker

import httpx
import re

load_dotenv()

# Configuration
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.index")
SQLITE_DB = os.getenv("SQLITE_DB", "metadata.sqlite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "300"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize embedder and FAISS index
embedder = LocalEmbedder(EMBED_MODEL)
sample_emb = embedder.encode(["hello"])
dim = sample_emb.shape[1]
index = FaissIndex(dim=dim, index_path=FAISS_INDEX_PATH, db_path=SQLITE_DB)

app = FastAPI(title="Enhanced Document Search & Chat System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class SearchQuery(BaseModel):
    q: str
    k: Optional[int] = 10
    page: Optional[int] = 1
    spellcheck: Optional[bool] = False
    search_type: Optional[str] = "hybrid"

class ChatMessage(BaseModel):
    message: str
    k: Optional[int] = 5
    include_context: Optional[bool] = True

class FileListRequest(BaseModel):
    query: str
    search_type: Optional[str] = "hybrid"

# Serve the main HTML interface
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html><head><title>Enhanced Document Search System</title></head>
        <body>
        <h1>Enhanced Document Search System</h1>
        <p>Please ensure index.html is in the project directory.</p>
        </body></html>
        """)

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "documents_indexed": get_document_count(),
        "chunks_indexed": get_chunk_count()
    }

def get_document_count():
    """Get total number of unique documents"""
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks")
        count = cur.fetchone()[0]
        conn.close()
        return count
    except:
        return 0

def get_chunk_count():
    """Get total number of chunks"""
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
        conn.close()
        return count
    except:
        return 0

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), spellcheck: bool = False):
    """Enhanced upload with stricter OCR cleaning"""
    try:
        file_path = UPLOAD_DIR / file.filename
        print(f"Processing upload: {file.filename}")

        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract text
        pages = extract_text(str(file_path))
        if not pages or all(p is None or not str(p).strip() for p in pages):
            raise HTTPException(status_code=400, detail="No text extracted from file.")

        print(f"Extracted {len(pages)} pages from {file.filename}")

        all_chunks, per_chunk_pages, per_chunk_offsets = [], [], []

        for page_num, page_text in enumerate(pages, start=1):
            if not page_text:
                continue

            print(f"Processing page {page_num}...")

            cleaned_page = clean_ocr_noise(str(page_text))
            cleaned_page = clean_text(cleaned_page, spellcheck=spellcheck)

            if not cleaned_page:
                print(f"Page {page_num} skipped (not meaningful)")
                continue

            # Intelligent chunking
            page_chunks = chunk_text_intelligently(
                cleaned_page,
                max_tokens=CHUNK_MAX_TOKENS,
                overlap_tokens=CHUNK_OVERLAP_TOKENS
            )

            # Only keep meaningful chunks
            page_chunks = [c for c in page_chunks if is_meaningful_text(c)]

            print(f"Page {page_num} split into {len(page_chunks)} clean chunks")

            for chunk_idx, chunk_text in enumerate(page_chunks):
                all_chunks.append(chunk_text.strip())
                per_chunk_pages.append(page_num)
                per_chunk_offsets.append(chunk_idx)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No valid text chunks after processing.")

        print(f"Total clean chunks: {len(all_chunks)}")

        embeddings = embed_texts(embedder, all_chunks)

        if embeddings.shape[0] != len(all_chunks):
            print(f"Warning: Embedding count mismatch. Expected {len(all_chunks)}, got {embeddings.shape[0]}")

        index.add_documents(
            doc_id=file.filename,
            file_path=str(file_path),
            texts=all_chunks,
            embeddings=embeddings,
            pages=per_chunk_pages,
            chunk_offsets=per_chunk_offsets
        )

        return {
            "message": f"Successfully uploaded and indexed {file.filename}",
            "chunks": len(all_chunks),
            "pages": len(pages),
            "filename": file.filename,
            "processing_details": {
                "original_pages": len(pages),
                "final_chunks": len(all_chunks),
                "embedding_dim": embeddings.shape[1] if embeddings.size > 0 else 0
            }
        }

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/search")
async def search_documents(
    query: Optional[str] = None,
    limit: int = 20,
    page: int = 1,
    search_type: str = "hybrid",  # "keyword", "semantic", or "hybrid"
    min_score: float = 0.0
):
    """Enhanced search with pagination and better ranking"""
    if not query or not query.strip():
        return {"items": [], "query": query, "total": 0, "page": page, "per_page": limit}

    query = query.strip()
    offset = (page - 1) * limit
    all_results = []
    seen_chunks = set()
    
    try:
        print(f"Search query: '{query}' | Type: {search_type} | Page: {page}")
        
        # --- Semantic search ---
        if search_type in ["semantic", "hybrid"]:
            print("Performing semantic search...")
            q_emb = embedder.encode([query])[0]
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
            
            # Get more results for better ranking
            sem_hits = index.search(q_emb, top_k=limit*5)
            
            print(f"Semantic search found {len(sem_hits)} results")
            
            for hit in sem_hits:
                if hit["chunk_id"] in seen_chunks:
                    continue
                
                # Filter by minimum score
                if hit.get("score", 0) < min_score:
                    continue

                text = hit["text"]
                
                # Enhanced highlighting for semantic results
                highlight_snippet = create_semantic_highlight(text, query)
                
                all_results.append({
                    "id": hit["chunk_id"],
                    "name": hit["doc_id"],
                    "page": hit.get("page", 1),
                    "file_path": hit.get("file_path", ""),
                    "text": text,
                    "highlighting": [{"lines": [highlight_snippet]}],
                    "score": hit.get("score", 0),
                    "search_type": "semantic",
                    "relevance_explanation": f"Semantic similarity: {hit.get('score', 0):.3f}"
                })
                seen_chunks.add(hit["chunk_id"])

        # --- Keyword search ---
        if search_type in ["keyword", "hybrid"]:
            print("Performing keyword search...")
            conn = sqlite3.connect(SQLITE_DB)
            cur = conn.cursor()
            
            # Enhanced FTS query
            search_terms = prepare_fts_query(query)
            
            cur.execute("""
                SELECT c.chunk_id, c.doc_id, c.file_path, c.page, c.chunk_index, c.text,
                       snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 50) as snippet,
                       rank
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (search_terms, limit*3))
            
            keyword_hits = cur.fetchall()
            conn.close()
            
            print(f"Keyword search found {len(keyword_hits)} results")

            for hit in keyword_hits:
                if hit[0] not in seen_chunks:
                    all_results.append({
                        "id": hit[0],
                        "name": hit[1],
                        "page": hit[3],
                        "file_path": hit[2],
                        "text": hit[5],
                        "highlighting": [{"lines": [hit[6]]}],
                        "score": 1.0 - (hit[7] / 100.0) if len(hit) > 7 else 1.0,  # Convert rank to score
                        "search_type": "keyword",
                        "relevance_explanation": "Keyword match"
                    })
                    seen_chunks.add(hit[0])

        # --- Enhanced ranking and pagination ---
        if search_type == "hybrid":
            # Combine and re-rank results
            all_results = rerank_hybrid_results(all_results, query)
        else:
            # Sort by score
            all_results.sort(key=lambda x: x["score"], reverse=True)

        # Get total count for pagination
        total_results = len(all_results)
        
        # Apply pagination
        paginated_results = all_results[offset:offset + limit]

        return {
            "items": paginated_results,
            "query": query,
            "total": total_results,
            "page": page,
            "per_page": limit,
            "total_pages": (total_results + limit - 1) // limit,
            "search_type": search_type,
            "has_more": offset + limit < total_results
        }

    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")



def prepare_fts_query(query: str) -> str:
    """Prepare query for FTS with better Arabic/English support"""
    # Remove special characters that might break FTS
    cleaned = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', query)
    words = cleaned.split()
    
    if not words:
        return f'"{query}"'
    
    # Create FTS query with both exact phrase and individual terms
    if len(words) > 1:
        exact_phrase = f'"{" ".join(words)}"'
        individual_terms = " OR ".join(words)
        return f'({exact_phrase}) OR ({individual_terms})'
    else:
        return words[0]




def create_semantic_highlight(text: str, query: str, max_length: int = 200) -> str:
    """Create highlighted snippet for semantic search results"""
    # Find the best snippet that might relate to the query
    sentences = re.split(r'[.!?]+', text)
    query_words = set(query.lower().split())
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        if len(sentence.strip()) < 20:
            continue
        
        sentence_words = set(sentence.lower().split())
        # Simple word overlap scoring
        overlap = len(query_words.intersection(sentence_words))
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence.strip()
    
    if not best_sentence:
        best_sentence = text[:max_length] + ("..." if len(text) > max_length else "")
    
    # Highlight query words
    for word in query_words:
        if len(word) > 2:  # Only highlight meaningful words
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            best_sentence = pattern.sub(f'<mark>{word}</mark>', best_sentence)
    
    return best_sentence



def rerank_hybrid_results(results: List[Dict], query: str) -> List[Dict]:
    """Advanced re-ranking for hybrid search results"""
    query_words = set(query.lower().split())
    
    for result in results:
        base_score = result["score"]
        
        # Boost for exact phrase matches
        if query.lower() in result["text"].lower():
            result["score"] *= 1.5
        
        # Boost for multiple query word matches
        text_words = set(result["text"].lower().split())
        word_overlap = len(query_words.intersection(text_words))
        if word_overlap > 1:
            result["score"] *= (1 + word_overlap * 0.1)
        
        # Boost for title/document name matches
        if any(word in result["name"].lower() for word in query_words):
            result["score"] *= 1.3
        
        # Penalty for very short texts
        if len(result["text"]) < 100:
            result["score"] *= 0.9
    
    return sorted(results, key=lambda x: x["score"], reverse=True)



@app.post("/files/list")
async def list_files_with_content(request: FileListRequest):
    """Get list of files and their pages containing specific content"""
    query = request.query.strip()
    if not query:
        return {"files": [], "query": query, "total_matches": 0}
    
    try:
        # Search for content
        search_results = await search_documents(
            query=query, 
            limit=100,  # Get more results for file listing
            search_type=request.search_type
        )
        
        # Group results by file
        files_dict = {}
        for item in search_results["items"]:
            file_name = item["name"]
            if file_name not in files_dict:
                files_dict[file_name] = {
                    "filename": file_name,
                    "file_path": item["file_path"],
                    "pages": [],
                    "total_matches": 0
                }
            
            # Add page info if not already present
            page_info = {
                "page": item["page"],
                "chunk_id": item["id"],
                "preview": item["text"][:200] + "..." if len(item["text"]) > 200 else item["text"],
                "score": item["score"],
                "highlighting": item.get("highlighting", [])
            }
            
            # Check if this page is already added
            existing_pages = [p["page"] for p in files_dict[file_name]["pages"]]
            if item["page"] not in existing_pages:
                files_dict[file_name]["pages"].append(page_info)
                files_dict[file_name]["total_matches"] += 1
            else:
                # Update existing page with better score if found
                for p in files_dict[file_name]["pages"]:
                    if p["page"] == item["page"] and item["score"] > p["score"]:
                        p.update(page_info)
        
        # Sort files by total matches and then by filename
        files_list = list(files_dict.values())
        files_list.sort(key=lambda x: (-x["total_matches"], x["filename"]))
        
        # Sort pages within each file
        for file_info in files_list:
            file_info["pages"].sort(key=lambda x: (-x["score"], x["page"]))
        
        return {
            "files": files_list,
            "query": query,
            "total_files": len(files_list),
            "total_matches": sum(f["total_matches"] for f in files_list),
            "search_type": request.search_type
        }
        
    except Exception as e:
        print(f"File listing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")




@app.get("/files/{filename}")
def serve_file(filename: str):
    """Serve uploaded files for viewing"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), media_type="application/pdf", filename=filename)




@app.get("/pdf/{item_id}")
async def view_pdf(item_id: int, highlight: Optional[str] = None):
    """Serve PDF with optional search highlighting"""
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cur = conn.cursor()
        cur.execute("SELECT file_path, page FROM chunks WHERE chunk_id = ?", (item_id,))
        result = cur.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
            
        file_path, page = result
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(file_path, media_type="application/pdf")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving PDF: {str(e)}")



@app.on_event("startup")
async def startup_event():
    await start_gpt4all_worker()
    print("Enhanced Document Search & Chat System started!")



@app.post("/chat")
async def chat_with_documents(message_data: ChatMessage):
    """Enhanced chat with better context retrieval and responses"""
    message = message_data.message.strip()
    k = message_data.k or 5

    if not message:
        return {"response": "Please provide a message.", "query": message}

    try:
        print(f"Chat query: '{message}'")
        
        # Special handling for file listing requests
        file_list_keywords = [
            "list files", "show files", "get files", "files with", "documents with",
            "which files", "what documents", "find files containing"
        ]
        
        if any(keyword in message.lower() for keyword in file_list_keywords):
            # Extract the search term from the message
            search_term = extract_search_term_from_message(message)
            if search_term:
                file_list_request = FileListRequest(query=search_term, search_type="hybrid")
                file_results = await list_files_with_content(file_list_request)
                
                # Format response
                if file_results["files"]:
                    response_parts = [f"Found {file_results['total_matches']} matches in {file_results['total_files']} files for '{search_term}':\n"]
                    
                    for file_info in file_results["files"][:10]:  # Limit to top 10 files
                        response_parts.append(f"\n📄 **{file_info['filename']}** ({file_info['total_matches']} matches)")
                        for page_info in file_info['pages'][:3]:  # Top 3 pages per file
                            response_parts.append(f"   • Page {page_info['page']}: {page_info['preview']}")
                    
                    if len(file_results["files"]) > 10:
                        response_parts.append(f"\n... and {len(file_results['files']) - 10} more files.")
                    
                    return {
                        "response": "".join(response_parts),
                        "query": message,
                        "files": file_results["files"][:10],
                        "search_term": search_term
                    }
                else:
                    return {
                        "response": f"No files found containing '{search_term}'.",
                        "query": message,
                        "search_term": search_term
                    }

        # Regular chat flow with enhanced context retrieval
        search_results = await search_documents(
            query=message, 
            limit=k*2,  # Get more results for better context
            search_type="hybrid"
        )
        
        retrieved_chunks = search_results.get("items", [])
        print(f"Retrieved {len(retrieved_chunks)} chunks for context")

        if not retrieved_chunks:
            return {
                "response": "I don't have any relevant information in the uploaded documents to answer your question. Please upload documents related to your query or ask about the content that has been uploaded.",
                "query": message
            }

        # Build enhanced context
        context_chunks = []
        relevant_documents = []
        
        for i, chunk in enumerate(retrieved_chunks[:k]):
            # Create rich context with metadata
            context_entry = f"""
Document: {chunk['name']}
Page: {chunk['page']}
Relevance Score: {chunk['score']:.3f}
Content: {chunk['text'][:400]}{'...' if len(chunk['text']) > 400 else ''}
---"""
            
            context_chunks.append(context_entry)
            relevant_documents.append({
                "id": chunk['id'],
                "name": chunk['name'],
                "page": chunk['page'],
                "file_path": chunk.get("file_path", ""),
                "score": chunk['score']
            })

        # Enhanced prompt for better responses
        context_text = "\n".join(context_chunks)
        
        enhanced_prompt = f"""
You are **ARCHIE**, a professional Document Intelligence Assistant. 
Your role is to help the user analyze, summarize, and retrieve knowledge 
from a collection of uploaded documents.

DOCUMENT EXCERPTS (from the knowledge base):

{context_text}

USER QUESTION: {message}

GUIDELINES FOR YOUR RESPONSE:
1. ✅ Base your answer ONLY on the provided documents. If the question is unrelated, politely explain that you can only assist with the uploaded documents.
2. 📑 If the user asks for a **summary** of a page or document:
   - Provide a clear **executive summary** in bullet points.
   - Highlight the **main themes, facts, and conclusions**.
   - Mention the **document name and page numbers**.
3. 📊 If the content involves structured data (lists, comparisons, steps, metrics):
   - Use a **well-formatted table** or **numbered list**.
4. 🔗 If the user asks about related or supporting documents:
   - Suggest other relevant documents from the excerpts.
   - Show relationships between them.
5. 🌍 If the document is in Arabic:
   - Provide the original text + an English translation (if useful).
6. ❌ If the answer cannot be found in the documents:
   - Say so clearly and advise the user to upload more relevant documents.

FORMAT RULES:
- Be concise, professional, and clear.
- Always **cite document names and page numbers** when making claims.
- Prefer bullet points or tables over long paragraphs.
- Start with a short **direct answer**, then add details if needed.

FINAL ANSWER:
"""

        response = await submit_prompt(enhanced_prompt[:4000])  # Limit prompt length
        
        # Clean up the response
        response = response.strip()
        if response.startswith("ANSWER:"):
            response = response[7:].strip()

        result = {
            "response": response,
            "query": message,
            "context_used": len(context_chunks)
        }
        
        if relevant_documents:
            result["relevant_documents"] = relevant_documents
            
        return result

    except Exception as e:
        print(f"Chat error: {e}")
        return {
            "response": f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question.",
            "query": message,
            "error": True
        }

def extract_search_term_from_message(message: str) -> str:
    """Extract the actual search term from a file listing request"""
    # Common patterns for file listing requests
    patterns = [
        r"files?\s+(?:that\s+)?(?:have|contain|with|having|containing)\s+['\"]?([^'\"]+)['\"]?",
        r"documents?\s+(?:that\s+)?(?:have|contain|with|having|containing)\s+['\"]?([^'\"]+)['\"]?",
        r"(?:list|show|find|get)\s+files?\s+.*?['\"]([^'\"]+)['\"]",
        r"(?:list|show|find|get)\s+files?\s+.*?\b(\w+(?:\s+\w+){0,3})\s*$",
        r"word\s+['\"]?([^'\"]+)['\"]?",
        r"term\s+['\"]?([^'\"]+)['\"]?",
    ]
    
    message_lower = message.lower()
    
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            term = match.group(1).strip()
            if len(term) > 1 and not term.lower() in ['of', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for']:
                return term
    
    # Fallback: look for quoted terms or the last meaningful words
    quoted_match = re.search(r'[\'\"](.*?)[\'\"]', message)
    if quoted_match:
        return quoted_match.group(1).strip()
    
    # Extract last few meaningful words as fallback
    words = message.lower().split()
    meaningful_words = [w for w in words if len(w) > 2 and w not in ['list', 'show', 'get', 'find', 'files', 'documents', 'with', 'containing', 'have', 'that']]
    
    if meaningful_words:
        return ' '.join(meaningful_words[-2:])  # Last 2 meaningful words
    
    return ""

# Legacy endpoints for compatibility
@app.get("/search/keyword")
def search_keyword_legacy(q: str, limit: int = 20):
    """Legacy keyword search endpoint"""
    return search_documents(query=q, limit=limit, search_type="keyword")

@app.post("/search/semantic")  
def search_semantic_legacy(query: SearchQuery):
    """Legacy semantic search endpoint"""
    return search_documents(query=query.q, limit=query.k or 5, search_type="semantic")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)