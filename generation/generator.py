"""
RAG generator.

Takes a query, retrieves relevant chunks, builds a prompt, and calls an LLM.
Returns the answer with source citations.
"""

import logging
from typing import Callable, Dict, List, Optional

from generation.llm.ollama_client import OllamaClient
from generation.prompts.builder import build_prompt
from retrieval.search.retriever import Retriever

logger = logging.getLogger(__name__)


class Generator:
    """
    RAG generator that orchestrates retrieval and generation.

    Usage:
        # With Qwen client
        generator = Generator(llm_client=QwenClient(api_key="your-api-key"))
        result = generator.generate("What are the symptoms of tuberculosis?")

        # Dry-run mode (returns prompt + sources without calling an LLM)
        generator = Generator()
        result = generator.generate("...")
    """

    def __init__(self, llm_client: Optional[Callable] = None):
        """
        Initialize the generator.

        Args:
            llm_client: Optional callable that takes a prompt string and returns
                        the LLM's response. If None, runs in dry-run mode
                        (returns the prompt + sources without calling an LLM).
        """
        self._retriever = Retriever()
        self._llm_client = llm_client

    def generate(self, query: str) -> Dict:
        """
        Generate an answer to the query.

        Args:
            query: The user's question.

        Returns:
            dict with keys:
                - "answer": str — the generated answer (or prompt if dry-run)
                - "sources": list[dict] — the retrieved chunks used
                - "llm_used": bool — whether an LLM was actually called
                - "llm_metadata": dict — additional LLM metadata (if applicable)
        """
        # ── Step 1: Retrieve chunks ────────────────────────────────────
        chunks = self._retriever.retrieve(query)
        if not chunks:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "sources": [],
                "llm_used": False,
                "llm_metadata": {},
            }

        # ── Step 2: Build prompt ───────────────────────────────────────
        prompt = build_prompt(query, chunks)

        # ── Step 3: Call LLM or return prompt (dry-run) ────────────────
        if self._llm_client:
            try:
                answer = self._llm_client.generate(prompt)

                llm_metadata = {
                    "model": getattr(self._llm_client, "model", "unknown"),
                }

                logger.info(f"Generated answer for query: '{query[:80]}...'")

                return {
                    "answer": answer,
                    "sources": chunks,
                    "llm_used": True,
                    "llm_metadata": llm_metadata,
                }

            except Exception as e:
                logger.error(f"LLM call failed: {e}")

                return {
                    "answer": f"Error calling LLM: {e}",
                    "sources": chunks,
                    "llm_used": False,
                    "llm_metadata": {},
                }
        else:
            logger.info(f"Dry-run mode: prompt generated for query: '{query[:80]}...'")
            return {
                "answer": prompt,
                "sources": chunks,
                "llm_used": False,
                "llm_metadata": {},
            }