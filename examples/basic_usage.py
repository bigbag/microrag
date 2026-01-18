"""Basic MicroRAG usage example.

This example demonstrates the core workflow:
- Creating a configuration
- Adding documents in various formats
- Building the index
- Searching for documents
- Managing documents (count, get, clear)

Before running, set MODEL_PATH to your sentence-transformer model directory.
"""

from microrag import RAGConfig, MicroRAG
from microrag.models import Document

# Set this to your sentence-transformer model path
MODEL_PATH = "/path/to/your/model"


def main() -> None:
    # Create configuration with in-memory database
    config = RAGConfig(
        model_path=MODEL_PATH,
        db_path=":memory:",
    )

    # Use context manager for automatic resource cleanup
    with MicroRAG(config) as rag:
        # Add documents - supports multiple formats

        # Format 1: Plain strings
        rag.add_documents([
            "Python is a high-level programming language.",
            "Machine learning enables computers to learn from data.",
        ])

        # Format 2: Dictionaries with metadata
        rag.add_documents([
            {
                "content": "DuckDB is an in-process analytical database.",
                "metadata": {"topic": "databases", "year": 2019},
            },
        ])

        # Format 3: Document objects
        rag.add_documents([
            Document(
                id="doc-vectors",
                content="Vector databases store embeddings for similarity search.",
                metadata={"topic": "databases"},
            ),
        ])

        # Build indexes (required before searching)
        print("Building index...")
        rag.build_index()

        # Search for documents
        print("\nSearching for 'database':")
        results = rag.search("database", top_k=3)
        for result in results:
            print(f"  [{result.rank}] Score: {result.score:.3f}")
            print(f"      Content: {result.content[:60]}...")
            if result.metadata:
                print(f"      Metadata: {result.metadata}")

        # Document management
        print(f"\nTotal documents: {rag.count()}")

        # Get specific document by ID
        doc = rag.get_document("doc-vectors")
        if doc:
            print(f"Retrieved doc: {doc.content[:50]}...")

        # Clear all documents
        rag.clear()
        print(f"After clear: {rag.count()} documents")


if __name__ == "__main__":
    main()
