"""
Metadata filter for retrieval results.

Provides post-retrieval filtering of chunks based on metadata fields
such as source_type, document_title, page_number, section, etc.

This module is designed to be called by Retriever after initial retrieval
(and optionally after reranking).
"""

import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MetadataFilter:
    """
    Filter retrieval results by metadata constraints.

    Usage:
        mf = MetadataFilter()
        # Filter to only NHP and ICMR sources
        filtered = mf.filter(results, source_types={"NHP", "ICMR"})
        # Filter to specific document
        filtered = mf.filter(results, document_titles={"NHP_Tuberculosis"})
        # Exclude certain sources
        filtered = mf.filter(results, exclude_source_types={"PubMed"})
    """

    # Known metadata keys that can be filtered on
    FILTERABLE_KEYS = {
        "source_type",
        "document_title",
        "source_file",
        "section",
        "page_number",
    }

    def filter(
        self,
        results: List[Dict],
        source_types: Optional[Set[str]] = None,
        document_titles: Optional[Set[str]] = None,
        source_files: Optional[Set[str]] = None,
        sections: Optional[Set[str]] = None,
        page_numbers: Optional[Set[int]] = None,
        exclude_source_types: Optional[Set[str]] = None,
        exclude_document_titles: Optional[Set[str]] = None,
        exclude_source_files: Optional[Set[str]] = None,
        exclude_sections: Optional[Set[str]] = None,
        exclude_page_numbers: Optional[Set[int]] = None,
    ) -> List[Dict]:
        """
        Filter results by metadata inclusion/exclusion criteria.

        All inclusion filters are ANDed together (a result must match ALL
        specified inclusion criteria). Exclusion filters remove any result
        that matches ANY exclusion criterion.

        Args:
            results: List of result dicts (each must have a "metadata" key).
            source_types: Only include results with these source_type values.
            document_titles: Only include results with these document_title values.
            source_files: Only include results with these source_file values.
            sections: Only include results with these section values.
            page_numbers: Only include results with these page_number values.
            exclude_source_types: Exclude results with these source_type values.
            exclude_document_titles: Exclude results with these document_title values.
            exclude_source_files: Exclude results with these source_file values.
            exclude_sections: Exclude results with these section values.
            exclude_page_numbers: Exclude results with these page_number values.

        Returns:
            Filtered list of result dicts.
        """
        if not results:
            return []

        filtered = []
        for result in results:
            metadata = result.get("metadata", {})

            # --- Inclusion checks (AND logic) ---
            if source_types is not None:
                if metadata.get("source_type") not in source_types:
                    continue
            if document_titles is not None:
                if metadata.get("document_title") not in document_titles:
                    continue
            if source_files is not None:
                if metadata.get("source_file") not in source_files:
                    continue
            if sections is not None:
                if metadata.get("section") not in sections:
                    continue
            if page_numbers is not None:
                if metadata.get("page_number") not in page_numbers:
                    continue

            # --- Exclusion checks (OR logic) ---
            if exclude_source_types is not None:
                if metadata.get("source_type") in exclude_source_types:
                    continue
            if exclude_document_titles is not None:
                if metadata.get("document_title") in exclude_document_titles:
                    continue
            if exclude_source_files is not None:
                if metadata.get("source_file") in exclude_source_files:
                    continue
            if exclude_sections is not None:
                if metadata.get("section") in exclude_sections:
                    continue
            if exclude_page_numbers is not None:
                if metadata.get("page_number") in exclude_page_numbers:
                    continue

            filtered.append(result)

        logger.info(
            "Metadata filter: %d → %d results "
            "(include: %s, exclude: %s)",
            len(results),
            len(filtered),
            self._describe_includes(
                source_types, document_titles, source_files, sections, page_numbers
            ),
            self._describe_excludes(
                exclude_source_types, exclude_document_titles,
                exclude_source_files, exclude_sections, exclude_page_numbers,
            ),
        )
        return filtered

    @staticmethod
    def _describe_includes(
        source_types, document_titles, source_files, sections, page_numbers
    ) -> str:
        parts = []
        if source_types:
            parts.append(f"source_type∈{source_types}")
        if document_titles:
            parts.append(f"title∈{document_titles}")
        if source_files:
            parts.append(f"file∈{source_files}")
        if sections:
            parts.append(f"section∈{sections}")
        if page_numbers:
            parts.append(f"page∈{page_numbers}")
        return " & ".join(parts) if parts else "none"

    @staticmethod
    def _describe_excludes(
        exclude_source_types, exclude_document_titles,
        exclude_source_files, exclude_sections, exclude_page_numbers,
    ) -> str:
        parts = []
        if exclude_source_types:
            parts.append(f"source_type∉{exclude_source_types}")
        if exclude_document_titles:
            parts.append(f"title∉{exclude_document_titles}")
        if exclude_source_files:
            parts.append(f"file∉{exclude_source_files}")
        if exclude_sections:
            parts.append(f"section∉{exclude_sections}")
        if exclude_page_numbers:
            parts.append(f"page∉{exclude_page_numbers}")
        return " & ".join(parts) if parts else "none"