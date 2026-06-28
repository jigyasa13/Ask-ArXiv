from sentence_transformers import CrossEncoder
from search import load_chunks, build_bm25_index, build_vector_index, hybrid_search

# -----------------------------------------------
# STEP A: Load the reranking model
# -----------------------------------------------
def load_reranker():
    """
    Load a cross-encoder model for reranking.
    
    Why this model?
    - cross-encoder/ms-marco-MiniLM-L-6-v2 is trained specifically
      to judge relevance between a question and a passage
    - Small enough to run on your laptop
    - Very accurate for our use case
    """
    print("⏳ Loading reranker model...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("✅ Reranker loaded")
    return reranker


# -----------------------------------------------
# STEP B: Rerank the 20 results
# -----------------------------------------------
def rerank_results(query, results, reranker, top_k=3):
    """
    Takes the 20 hybrid search results and picks the best top_k.

    How it works:
    - Creates pairs of (question, chunk) for every result
    - Cross-encoder reads each pair together and gives a score
    - Higher score = more relevant
    - We sort by score and return top_k

    Why top_k=3?
    - 3 chunks is enough context for GPT to answer well
    - Sending more would increase cost and response time
    - Sending less might miss important information
    """
    if not results:
        print("❌ No results to rerank")
        return []

    # Create question-chunk pairs
    pairs = [(query, result["chunk_text"]) for result in results]

    # Score each pair
    print(f"⏳ Reranking {len(pairs)} results...")
    scores = reranker.predict(pairs)

    # Attach scores to results
    for i, result in enumerate(results):
        result["rerank_score"] = float(scores[i])

    # Sort by rerank score (highest first)
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    # Return only top_k
    top_results = reranked[:top_k]

    print(f"✅ Top {top_k} results selected")
    return top_results


# -----------------------------------------------
# STEP C: Pretty print results (for testing)
# -----------------------------------------------
def print_results(results):
    """
    Just for testing — prints results in a readable way.
    """
    for i, result in enumerate(results):
        print(f"\n{'='*50}")
        print(f"Result {i+1}")
        print(f"{'='*50}")
        print(f"📄 Title:  {result['title']}")
        print(f"👤 Authors: {', '.join(result['authors'][:2])}")
        print(f"📅 Year:   {result['year']}")
        print(f"🔗 URL:    {result['url']}")
        print(f"⭐ Rerank Score: {result['rerank_score']:.4f}")
        print(f"\n📝 Relevant Text:\n{result['chunk_text'][:300]}...")


# -----------------------------------------------
# MAIN — Test everything
# -----------------------------------------------
if __name__ == "__main__":
    # Load everything from Week 3
    print("🚀 Loading data and indexes...\n")
    chunks = load_chunks()
    bm25 = build_bm25_index(chunks)
    index, model, embeddings = build_vector_index(chunks)

    # Load reranker
    reranker = load_reranker()

    # Test question
    query = "How does attention work in transformers?"
    print(f"\n🔍 Question: '{query}'\n")

    # Step 1: Hybrid search → 20 results
    results = hybrid_search(query, bm25, index, model, chunks)

    # Step 2: Rerank → top 3
    top_3 = rerank_results(query, results, reranker)

    # Print final results
    print("\n🏆 Final Top 3 Results After Reranking:")
    print_results(top_3)