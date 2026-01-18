"""FAQ search example with MicroRAG.

This example demonstrates a practical use case:
- Loading FAQ entries with metadata (category, source)
- Configuring for FAQ-style short documents (disable chunking)
- Searching with different queries
- Filtering and displaying results with metadata

Before running, set MODEL_PATH to your sentence-transformer model directory.
"""

from microrag import RAGConfig, MicroRAG

# Set this to your sentence-transformer model path
MODEL_PATH = "/path/to/your/model"

# Sample FAQ data with categories
FAQ_ENTRIES = [
    {
        "content": "To reset your password, click 'Forgot Password' on the login page "
        "and follow the email instructions.",
        "metadata": {"category": "account", "id": "faq-001"},
    },
    {
        "content": "Free shipping is available on orders over $50. Standard delivery "
        "takes 3-5 business days.",
        "metadata": {"category": "shipping", "id": "faq-002"},
    },
    {
        "content": "You can return items within 30 days of purchase. Items must be "
        "unused and in original packaging.",
        "metadata": {"category": "returns", "id": "faq-003"},
    },
    {
        "content": "We accept Visa, MasterCard, American Express, and PayPal for payments.",
        "metadata": {"category": "payments", "id": "faq-004"},
    },
    {
        "content": "To track your order, log into your account and visit the "
        "'Order History' section.",
        "metadata": {"category": "shipping", "id": "faq-005"},
    },
    {
        "content": "Gift cards can be purchased in denominations of $25, $50, or $100 "
        "and never expire.",
        "metadata": {"category": "payments", "id": "faq-006"},
    },
    {
        "content": "To update your email address, go to Account Settings and click "
        "'Edit Profile'.",
        "metadata": {"category": "account", "id": "faq-007"},
    },
    {
        "content": "Refunds are processed within 5-7 business days after we receive "
        "your returned item.",
        "metadata": {"category": "returns", "id": "faq-008"},
    },
]


def main() -> None:
    # Configure for FAQ-style short documents
    config = RAGConfig(
        model_path=MODEL_PATH,
        db_path=":memory:",
        chunk_size=2000,  # Large enough to avoid chunking FAQs
        similarity_threshold=0.3,
    )

    with MicroRAG(config) as rag:
        # Add FAQ entries (chunk=False for short documents)
        rag.add_documents(FAQ_ENTRIES, chunk=False)
        rag.build_index()

        print(f"Loaded {rag.count()} FAQ entries\n")

        # Example searches
        queries = [
            "How do I change my password?",
            "What's your return policy?",
            "How long does delivery take?",
        ]

        for query in queries:
            print(f"Q: {query}")
            results = rag.search(query, top_k=2)

            if not results:
                print("  No matching FAQs found.\n")
                continue

            for result in results:
                category = result.metadata.get("category", "unknown")
                faq_id = result.metadata.get("id", "unknown")
                print(f"  [{category.upper()}] ({faq_id}) Score: {result.score:.3f}")
                print(f"  A: {result.content}\n")

        # Filter results by category (post-search filtering)
        print("=== Filtering by category ===")
        print("All results for 'payment options', filtered to 'payments' category:")
        all_results = rag.search("payment options", top_k=5)
        payment_results = [
            r for r in all_results if r.metadata.get("category") == "payments"
        ]
        for result in payment_results:
            print(f"  [{result.metadata.get('id')}] {result.content}")


if __name__ == "__main__":
    main()
