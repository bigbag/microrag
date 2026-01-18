"""Query preprocessing for MicroRAG."""

import re
from collections.abc import Sequence

from microrag.stopwords import ENGLISH_STOPWORDS


class QueryProcessor:
    """Query processor for normalization, expansion, and tokenization.

    This class handles:
    - Whitespace normalization
    - Abbreviation expansion
    - Tokenization for BM25 with stopword removal

    Args:
        abbreviations: Mapping of abbreviations to their expansions.
        stopwords: Set of stopwords to remove during tokenization.
        remove_stopwords: Whether to remove stopwords during tokenization.
    """

    # Pattern to match word boundaries for abbreviation replacement
    _WORD_PATTERN = re.compile(r"\b([A-Za-z0-9]+)\b")
    # Pattern for tokenization (alphanumeric sequences)
    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

    def __init__(
        self,
        abbreviations: dict[str, str] | None = None,
        stopwords: frozenset[str] | None = None,
        remove_stopwords: bool = True,
    ) -> None:
        self._abbreviations = abbreviations or {}
        self._stopwords = stopwords if stopwords is not None else ENGLISH_STOPWORDS
        self._remove_stopwords = remove_stopwords

        # Build case-insensitive abbreviation lookup
        self._abbrev_lookup: dict[str, str] = {}
        for abbrev, expansion in self._abbreviations.items():
            self._abbrev_lookup[abbrev.lower()] = expansion

    def normalize(self, text: str) -> str:
        """Normalize text by collapsing whitespace.

        Args:
            text: Input text.

        Returns:
            Normalized text with collapsed whitespace.
        """
        return " ".join(text.split())

    def expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations in text.

        Args:
            text: Input text with potential abbreviations.

        Returns:
            Text with abbreviations expanded.
        """
        if not self._abbreviations:
            return text

        def replace_abbrev(match: re.Match[str]) -> str:
            word = match.group(1)
            lower_word = word.lower()
            if lower_word in self._abbrev_lookup:
                return self._abbrev_lookup[lower_word]
            return word

        return self._WORD_PATTERN.sub(replace_abbrev, text)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 search.

        Args:
            text: Input text.

        Returns:
            List of lowercase tokens with stopwords optionally removed.
        """
        lower_text = text.lower()
        tokens = self._TOKEN_PATTERN.findall(lower_text)

        if self._remove_stopwords and self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        return tokens

    def process(self, query: str) -> str:
        """Process query: normalize and expand abbreviations.

        This is the main method for preparing queries for semantic search.

        Args:
            query: Raw query string.

        Returns:
            Processed query ready for embedding.
        """
        normalized = self.normalize(query)
        expanded = self.expand_abbreviations(normalized)
        return expanded

    def process_for_bm25(self, query: str) -> list[str]:
        """Process query for BM25 search.

        Args:
            query: Raw query string.

        Returns:
            List of tokens for BM25 matching.
        """
        processed = self.process(query)
        return self.tokenize(processed)

    def tokenize_documents(self, documents: Sequence[str]) -> list[list[str]]:
        """Tokenize multiple documents for BM25 indexing.

        Args:
            documents: Sequence of document texts.

        Returns:
            List of token lists, one per document.
        """
        return [self.tokenize(doc) for doc in documents]
