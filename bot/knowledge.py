import asyncio
import logging
import os
import re
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

_client: Optional[chromadb.ClientAPI] = None
_collection = None
_embedder = None


def _get_chroma_client(chroma_dir: str):
    global _client
    if _client is None:
        os.makedirs(chroma_dir, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
    return _client


def _get_collection(chroma_dir: str):
    global _collection
    if _collection is None:
        client = _get_chroma_client(chroma_dir)
        _collection = client.get_or_create_collection(
            name="java_books",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _load_embedder(model_name: str):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _embedder = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")
    return _embedder


def _extract_pdf_text(pdf_path: str) -> list[dict]:
    """Extract text from PDF, return list of {text, page, title, author}"""
    from pypdf import PdfReader
    chunks_info = []
    try:
        reader = PdfReader(pdf_path)
        meta = reader.metadata or {}
        title = meta.title or os.path.splitext(os.path.basename(pdf_path))[0]
        author = meta.author or "Unknown"

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                chunks_info.append({
                    "text": text,
                    "page": page_num,
                    "title": title,
                    "author": author,
                })
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", pdf_path, e)
    return chunks_info


def _split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks"""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


async def index_knowledge_base(knowledge_root: str, chroma_dir: str, embedding_model: str,
                                chunk_size: int = 500, chunk_overlap: int = 50) -> tuple[int, int]:
    """Index all PDFs from knowledge_root. Returns (book_count, chunk_count)."""

    def _do_indexing():
        embedder = _load_embedder(embedding_model)
        client = _get_chroma_client(chroma_dir)

        # Delete and recreate collection for fresh indexing
        try:
            client.delete_collection("java_books")
        except Exception:
            pass

        global _collection
        _collection = client.create_collection(
            name="java_books",
            metadata={"hnsw:space": "cosine"}
        )

        book_count = 0
        chunk_count = 0

        if not os.path.exists(knowledge_root):
            logger.warning("Knowledge root does not exist: %s", knowledge_root)
            return 0, 0

        for root, dirs, files in os.walk(knowledge_root):
            for fname in files:
                if not fname.lower().endswith(".pdf"):
                    continue
                pdf_path = os.path.join(root, fname)
                logger.info("Indexing: %s", pdf_path)
                try:
                    pages = _extract_pdf_text(pdf_path)
                    if not pages:
                        continue

                    book_count += 1
                    doc_ids = []
                    doc_texts = []
                    doc_embeddings = []
                    doc_metadatas = []

                    for page_info in pages:
                        page_chunks = _split_into_chunks(page_info["text"], chunk_size, chunk_overlap)
                        for i, chunk_text in enumerate(page_chunks):
                            chunk_id = f"{fname}__p{page_info['page']}__c{i}"
                            doc_ids.append(chunk_id)
                            doc_texts.append(chunk_text)
                            doc_metadatas.append({
                                "book_title": page_info["title"],
                                "author": page_info["author"],
                                "page": str(page_info["page"]),
                                "file": fname,
                            })

                    if doc_texts:
                        embeddings = embedder.encode(doc_texts, show_progress_bar=False).tolist()
                        batch_size = 500
                        for b in range(0, len(doc_ids), batch_size):
                            _collection.add(
                                ids=doc_ids[b:b+batch_size],
                                documents=doc_texts[b:b+batch_size],
                                embeddings=embeddings[b:b+batch_size],
                                metadatas=doc_metadatas[b:b+batch_size],
                            )
                        chunk_count += len(doc_ids)
                        logger.info("  → %d chunks from %s", len(doc_ids), fname)

                except Exception as e:
                    logger.error("Error indexing %s: %s", pdf_path, e)

        return book_count, chunk_count

    return await asyncio.to_thread(_do_indexing)


async def search_knowledge(query: str, chroma_dir: str, embedding_model: str, top_k: int = 5) -> list[dict]:
    """Search ChromaDB for relevant chunks. Returns list of {text, metadata}."""

    def _do_search():
        embedder = _load_embedder(embedding_model)
        collection = _get_collection(chroma_dir)

        count = collection.count()
        if count == 0:
            return []

        query_emb = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_emb,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({"text": doc, "metadata": meta, "distance": dist})
        return chunks

    return await asyncio.to_thread(_do_search)


async def clear_knowledge_base(chroma_dir: str):
    """Delete all chunks from ChromaDB."""

    def _do_clear():
        global _collection
        client = _get_chroma_client(chroma_dir)
        try:
            client.delete_collection("java_books")
            _collection = None
            logger.info("Knowledge base cleared")
        except Exception as e:
            logger.error("Error clearing knowledge base: %s", e)

    await asyncio.to_thread(_do_clear)


async def get_chunk_count(chroma_dir: str) -> int:
    def _count():
        try:
            collection = _get_collection(chroma_dir)
            return collection.count()
        except Exception:
            return 0
    return await asyncio.to_thread(_count)
