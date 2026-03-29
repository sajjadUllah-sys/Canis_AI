"""
rag/indexer.py

One-time pipeline to:
1. Extract text from all PDFs in the knowledge base folder
2. Detect language of each chunk
3. Translate Spanish (or other non-English) chunks to English via OpenAI
4. Chunk text into ~400 token passages
5. Embed with text-embedding-3-small
6. Store in ChromaDB with metadata

Run this ONCE before the app goes live, or re-run whenever knowledge files change.
Usage:
    python -m canis.rag.indexer --pdf_dir ./knowledge_base --db_dir ./chroma_db
"""

import os
import re
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

import chromadb
import tiktoken
from langdetect import detect, LangDetectException
from openai import OpenAI
from pypdf import PdfReader

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────
EMBEDDING_MODEL     = "text-embedding-3-small"
TRANSLATION_MODEL   = "gpt-4o-mini"          # cheap + fast for translation
CHUNK_SIZE_TOKENS   = 400
CHUNK_OVERLAP_TOKENS = 50
COLLECTION_NAME     = "canis_knowledge"


# ══════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        log.error(f"Failed to extract text from {pdf_path.name}: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS,
               overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """
    Split text into overlapping token-based chunks.
    Tries to split on paragraph/sentence boundaries when possible.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        # Try to end at a sentence boundary
        last_period = max(
            chunk_text.rfind(". "),
            chunk_text.rfind(".\n"),
            chunk_text.rfind("? "),
            chunk_text.rfind("! "),
        )
        if last_period > chunk_size * 0.6 * 4:  # rough char estimate
            chunk_text = chunk_text[:last_period + 1]

        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


# ══════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION & TRANSLATION
# ══════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Detect language of a text chunk.
    Returns ISO 639-1 code (e.g. 'en', 'es').
    Returns 'en' if detection fails.
    """
    try:
        sample = text[:500]  # only need a sample for detection
        return detect(sample)
    except LangDetectException:
        return "en"


def translate_to_english(text: str, client: OpenAI) -> str:
    """
    Translate non-English text to English using OpenAI.
    Used during indexing only — not at query time.
    """
    try:
        response = client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator specializing in "
                        "animal behavior and veterinary science. "
                        "Translate the following text to English. "
                        "Preserve technical terminology accurately. "
                        "Output only the translated text, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"Translation failed: {e}")
        return text  # fallback: store original if translation fails


# ══════════════════════════════════════════════════════════════════
# EMBEDDING
# ══════════════════════════════════════════════════════════════════

def embed_chunks(chunks: list[str], client: OpenAI) -> list[list[float]]:
    """
    Embed a list of text chunks using OpenAI's embedding model.
    Batches requests to stay within API limits.
    """
    embeddings = []
    batch_size = 100  # OpenAI allows up to 2048 inputs but 100 is safe

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            log.info(f"  Embedded batch {i//batch_size + 1} ({len(batch)} chunks)")
            time.sleep(0.5)  # small delay to avoid rate limits
        except Exception as e:
            log.error(f"Embedding batch failed: {e}")
            # Fill with empty embeddings so indexing doesn't halt
            embeddings.extend([[] for _ in batch])

    return embeddings


# ══════════════════════════════════════════════════════════════════
# MAIN INDEXING PIPELINE
# ══════════════════════════════════════════════════════════════════

def build_index(pdf_dir: str, db_dir: str, api_key: str, translate: bool = False) -> None:
    """
    Full indexing pipeline.
    Reads all PDFs from pdf_dir, processes them, stores in ChromaDB at db_dir.
    """
    client = OpenAI(api_key=api_key)

    # ── Setup ChromaDB ────────────────────────────────────────────
    chroma_client = chromadb.PersistentClient(path=db_dir)

    # Delete existing collection if re-indexing
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        log.info("Deleted existing collection for re-indexing.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # ── Process each PDF ──────────────────────────────────────────
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    log.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    log.info(f"Translation: {'ENABLED' if translate else 'DISABLED (multilingual embeddings)'}")

    total_chunks = 0
    global_chunk_id = 0

    for file_num, pdf_path in enumerate(pdf_files, 1):
        log.info(f"\n[{file_num}/{len(pdf_files)}] Processing: {pdf_path.name}")

        # 1. Extract text
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            log.warning(f"  No text extracted from {pdf_path.name} — skipping.")
            continue

        # 2. Chunk
        chunks = chunk_text(raw_text)
        log.info(f"  → {len(chunks)} chunks")

        # 3. Detect language & optionally translate
        chunk_languages = [detect_language(chunk) for chunk in chunks]
        if translate:
            processed_chunks = []
            for chunk, lang in zip(chunks, chunk_languages):
                if lang != "en":
                    log.info(f"  Translating chunk (detected: {lang}) ...")
                    chunk = translate_to_english(chunk, client)
                processed_chunks.append(chunk)
        else:
            non_en = sum(1 for l in chunk_languages if l != "en")
            if non_en:
                log.info(f"  {non_en}/{len(chunks)} non-English chunks (skipping translation)")
            processed_chunks = chunks

        # 4. Embed
        log.info(f"  Embedding {len(processed_chunks)} chunks ...")
        embeddings = embed_chunks(processed_chunks, client)

        # 5. Store in ChromaDB
        valid_chunks    = []
        valid_embeddings = []
        valid_ids       = []
        valid_metadatas = []

        for i, (chunk, embedding) in enumerate(zip(processed_chunks, embeddings)):
            if not embedding:
                continue
            chunk_id = f"chunk_{global_chunk_id}"
            valid_ids.append(chunk_id)
            valid_chunks.append(chunk)
            valid_embeddings.append(embedding)
            lang = chunk_languages[i]
            valid_metadatas.append({
                "source_file": pdf_path.name,
                "chunk_index": i,
                "original_language": lang,
                "translated": translate and lang != "en",
            })
            global_chunk_id += 1

        if valid_chunks:
            collection.add(
                ids=valid_ids,
                documents=valid_chunks,
                embeddings=valid_embeddings,
                metadatas=valid_metadatas,
            )
            total_chunks += len(valid_chunks)
            log.info(f"  ✓ Stored {len(valid_chunks)} chunks from {pdf_path.name}")

    log.info(f"\n{'='*50}")
    log.info(f"Indexing complete.")
    log.info(f"Total chunks stored: {total_chunks}")
    log.info(f"Collection: '{COLLECTION_NAME}' at {db_dir}")
    log.info(f"{'='*50}")


# ── CLI Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CANIS knowledge index")
    parser.add_argument("--pdf_dir", required=True, help="Path to folder containing PDFs")
    parser.add_argument("--db_dir",  required=True, help="Path to ChromaDB storage folder")
    parser.add_argument("--api_key", default=None,  help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--translate", action="store_true", help="Translate non-English chunks to English (slow, usually not needed)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required. Pass --api_key or set OPENAI_API_KEY.")

    build_index(
        pdf_dir=args.pdf_dir,
        db_dir=args.db_dir,
        api_key=key,
        translate=args.translate,
    )
