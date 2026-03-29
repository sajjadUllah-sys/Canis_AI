"""
rag/retriever.py

Query-time retrieval from ChromaDB.
Embeds the user's message and returns the top-K most relevant
knowledge chunks to inject into the system prompt as context.
"""

import os
import logging
from typing import Optional

import chromadb
from openai import OpenAI

log = logging.getLogger(__name__)

EMBEDDING_MODEL  = "text-embedding-3-small"
COLLECTION_NAME  = "canis_knowledge"
DEFAULT_TOP_K    = 5


class Retriever:
    """
    Handles query-time RAG retrieval from ChromaDB.
    Instantiate once and reuse across requests.
    """

    def __init__(self, db_dir: str, api_key: str):
        """
        Args:
            db_dir:  Path to the ChromaDB folder created by indexer.py
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.chroma = chromadb.PersistentClient(path=db_dir)
        self.collection = self.chroma.get_collection(COLLECTION_NAME)
        log.info(f"Retriever ready. Collection has {self.collection.count()} chunks.")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K,
                 dog_profile: Optional[dict] = None) -> list[dict]:
        """
        Retrieve the top-K most relevant knowledge chunks for a query.

        Args:
            query:       The user's message
            top_k:       Number of chunks to return
            dog_profile: Optional profile dict — used to enrich the query
                         with breed/condition keywords for better retrieval

        Returns:
            List of dicts: [{"text": ..., "source": ..., "score": ...}, ...]
        """

        # Enrich query with profile context for better retrieval
        enriched_query = self._enrich_query(query, dog_profile)

        # Embed the query
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=enriched_query
        )
        query_embedding = response.data[0].embedding

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        chunks = []
        documents  = results["documents"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert cosine distance to similarity score (0-1)
            similarity = 1 - dist
            chunks.append({
                "text":       doc,
                "source":     meta.get("source_file", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "translated": meta.get("translated", False),
                "score":      round(similarity, 4),
            })

        # Filter out very low relevance chunks (score < 0.3)
        chunks = [c for c in chunks if c["score"] >= 0.3]

        log.debug(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}...'")
        return chunks

    def format_context_block(self, chunks: list[dict]) -> str:
        """
        Formats retrieved chunks into a clean context block
        for injection into the system prompt.

        Args:
            chunks: Output from retrieve()

        Returns:
            Formatted string to inject between profile and conversation
        """
        if not chunks:
            return ""

        lines = ["--- KNOWLEDGE CONTEXT ---"]
        lines.append("The following information from the knowledge base is relevant to this query.")
        lines.append("Use it to inform your response, but always apply CANIS guidelines first.\n")

        for i, chunk in enumerate(chunks, 1):
            source = chunk["source"].replace(".pdf", "").replace("_", " ")
            lines.append(f"[Source {i}: {source}]")
            lines.append(chunk["text"])
            lines.append("")

        lines.append("--- END OF KNOWLEDGE CONTEXT ---")
        return "\n".join(lines)

    def _enrich_query(self, query: str, dog_profile: Optional[dict]) -> str:
        """
        Appends key profile terms to the query to improve retrieval.
        E.g., breed, medical conditions, behavioral conditions.
        """
        if not dog_profile:
            return query

        enrichment_parts = []

        breed = dog_profile.get("breed")
        if breed:
            enrichment_parts.append(breed)

        medical = dog_profile.get("medical_conditions", [])
        if medical:
            enrichment_parts.extend(medical)

        behavioral = dog_profile.get("behavioral_conditions", [])
        if behavioral:
            enrichment_parts.extend(behavioral)

        if enrichment_parts:
            return f"{query} [{' '.join(enrichment_parts)}]"

        return query


# ── Standalone test ───────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    retriever = Retriever(
        db_dir="./chroma_db",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    test_profile = {
        "breed": "Golden Retriever",
        "medical_conditions": ["Anxiety", "Hip Dysplasia"],
        "behavioral_conditions": ["Reactivity", "Barking Issues"],
    }

    results = retriever.retrieve(
        query="Why does my dog bark at other dogs on walks?",
        top_k=4,
        dog_profile=test_profile
    )

    print(retriever.format_context_block(results))
