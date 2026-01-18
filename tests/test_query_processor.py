"""Tests for QueryProcessor."""

from microrag.query_processor import QueryProcessor


class TestQueryProcessor:
    """Tests for QueryProcessor class."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        qp = QueryProcessor()
        assert qp.normalize("  hello   world  ") == "hello world"
        assert qp.normalize("hello\n\nworld") == "hello world"

    def test_expand_abbreviations(self):
        """Test abbreviation expansion."""
        qp = QueryProcessor(
            abbreviations={"ML": "machine learning", "NLP": "natural language processing"}
        )
        assert qp.expand_abbreviations("ML techniques") == "machine learning techniques"
        assert (
            qp.expand_abbreviations("NLP and ML")
            == "natural language processing and machine learning"
        )
        # Case insensitive
        assert qp.expand_abbreviations("ml techniques") == "machine learning techniques"
        # Word boundaries respected
        assert qp.expand_abbreviations("HTML") == "HTML"

    def test_tokenize(self):
        """Test tokenization with and without stopwords."""
        qp = QueryProcessor(remove_stopwords=True)
        tokens = qp.tokenize("The quick brown fox is jumping")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "quick" in tokens

        qp_no_stop = QueryProcessor(remove_stopwords=False)
        tokens = qp_no_stop.tokenize("The quick brown fox")
        assert "the" in tokens

    def test_process_for_bm25(self):
        """Test BM25 query processing with abbreviations."""
        qp = QueryProcessor(
            abbreviations={"ML": "machine learning"},
            remove_stopwords=True,
        )
        tokens = qp.process_for_bm25("What is ML?")
        assert "machine" in tokens
        assert "learning" in tokens
        assert "what" not in tokens
        assert "is" not in tokens
