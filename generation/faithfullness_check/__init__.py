"""
Faithfulness / hallucination check module.

Runs a post-generation verification step that uses an LLM-as-verifier
to check whether each factual claim in the generated answer is supported
by the retrieved source chunks. When the check fails, a fallback response
is returned instead of the unverified answer.
"""