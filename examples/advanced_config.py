"""Advanced MicroRAG configuration example.

This example demonstrates:
- Custom abbreviations for domain-specific terms
- Hybrid search tuning (alpha weight, threshold)
- Chunk size configuration for long documents
- Semantic-only vs hybrid search comparison
- Using config.with_updates() for config variants

Before running, set MODEL_PATH to your sentence-transformer model directory.
"""

from microrag import RAGConfig, MicroRAG

# Set this to your sentence-transformer model path
MODEL_PATH = "/path/to/your/model"


def main() -> None:
    # Advanced configuration with custom settings
    config = RAGConfig(
        model_path=MODEL_PATH,
        db_path=":memory:",
        # Custom abbreviations - queries with "ML" will expand to "machine learning"
        abbreviations={
            "ML": "machine learning",
            "NLP": "natural language processing",
            "LLM": "large language model",
            "RAG": "retrieval augmented generation",
        },
        # Hybrid search settings
        hybrid_enabled=True,
        hybrid_alpha=0.7,  # 70% semantic, 30% keyword (BM25+FTS)
        similarity_threshold=0.3,  # Lower threshold to include more results
        # Chunking for long documents
        chunk_size=500,  # Smaller chunks for more granular retrieval
        chunk_overlap=100,
    )

    print("=== Configuration ===")
    print(f"Hybrid alpha: {config.hybrid_alpha}")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Abbreviations: {config.abbreviations}")

    # Sample documents about AI/ML topics
    documents = [
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn and improve from experience without being explicitly programmed.",
        "Natural language processing allows computers to understand, interpret, "
        "and generate human language in useful ways.",
        "Large language models like GPT and Claude are trained on vast text corpora "
        "to generate human-like text and perform various language tasks.",
        "Retrieval augmented generation combines information retrieval with "
        "language generation to produce more accurate and grounded responses.",
        "Deep learning uses neural networks with many layers to learn complex "
        "patterns in data, enabling breakthroughs in computer vision and NLP.",
    ]

    with MicroRAG(config) as rag:
        rag.add_documents(documents)
        rag.build_index()

        # Search with abbreviation expansion
        # "ML" will be expanded to "machine learning" during query processing
        print("\n=== Abbreviation Expansion ===")
        print("Query: 'ML applications'")
        results = rag.search("ML applications", top_k=2)
        for r in results:
            print(f"  [{r.score:.3f}] {r.content[:70]}...")

        # Compare hybrid vs semantic-only search
        print("\n=== Hybrid vs Semantic Search ===")
        query = "neural networks deep learning"

        print(f"Query: '{query}'")

        print("\nHybrid search (semantic + BM25 + FTS):")
        hybrid_results = rag.search(query, top_k=2, hybrid=True)
        for r in hybrid_results:
            print(f"  [{r.score:.3f}] {r.content[:70]}...")

        print("\nSemantic-only search:")
        semantic_results = rag.search(query, top_k=2, hybrid=False)
        for r in semantic_results:
            print(f"  [{r.score:.3f}] {r.content[:70]}...")

        # Adjusting threshold at search time
        print("\n=== Threshold Comparison ===")
        print("Query: 'quantum computing' (not in documents)")

        low_threshold_results = rag.search("quantum computing", top_k=3, threshold=0.1)
        print(f"Threshold 0.1: {len(low_threshold_results)} results")

        high_threshold_results = rag.search("quantum computing", top_k=3, threshold=0.5)
        print(f"Threshold 0.5: {len(high_threshold_results)} results")

    # Create config variant using with_updates()
    print("\n=== Config Variants with with_updates() ===")
    strict_config = config.with_updates(
        similarity_threshold=0.6,
        hybrid_alpha=0.9,  # More weight on semantic search
    )
    print(f"Original threshold: {config.similarity_threshold}")
    print(f"Strict threshold: {strict_config.similarity_threshold}")
    print(f"Original alpha: {config.hybrid_alpha}")
    print(f"Strict alpha: {strict_config.hybrid_alpha}")


if __name__ == "__main__":
    main()
