import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# -----------------------------------------------
# STEP A: Load chunks we saved in Week 2
# -----------------------------------------------
def load_chunks():
    """
    Load all chunks from the JSON file we created in Week 2.
    Why: We need the chunks to build our search indexes.
    """
    with open("../data/chunks.json", "r") as f:
        chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks")
    return chunks


# -----------------------------------------------
# STEP B: Build BM25 Index (keyword search)
# -----------------------------------------------
def build_bm25_index(chunks):
    """
    BM25 works on words. So we split every chunk into a list of words.
    Why: BM25 needs tokenized (word by word) text to work.
    """
    tokenized_chunks = [chunk["chunk_text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    print("✅ BM25 index built")
    return bm25


# -----------------------------------------------
# STEP C: Build Vector Index (meaning search)
# -----------------------------------------------
def build_vector_index(chunks):
    """
    Convert every chunk into a vector (list of numbers) using a
    pre-trained model. Then store in FAISS for fast searching.
    Why: Vectors capture the MEANING of text, not just keywords.
    """
    print("⏳ Building vector index... (this takes 1-2 mins first time)")

    # Load a pre-trained model that converts text to vectors
    model = SentenceTransformer("../models/cs-embedding-model")

    # Convert all chunks to vectors
    texts = [chunk["chunk_text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Store vectors in FAISS
    dimension = embeddings.shape[1]  # size of each vector (384 numbers)
    index = faiss.IndexFlatL2(dimension)  # L2 = measures distance between vectors
    index.add(np.array(embeddings))

    print("✅ Vector index built")
    return index, model, embeddings


# -----------------------------------------------
# STEP D: BM25 Search
# -----------------------------------------------
def bm25_search(query, bm25, chunks, top_k=10):
    """
    Search using keywords.
    Returns top_k most relevant chunks.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get indexes of top scoring chunks
    top_indexes = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indexes:
        results.append({
            "chunk_text": chunks[idx]["chunk_text"],
            "title": chunks[idx]["title"],
            "authors": chunks[idx]["authors"],
            "year": chunks[idx]["year"],
            "url": chunks[idx]["url"],
            "score": float(scores[idx]),
            "search_type": "bm25"
        })
    return results


# -----------------------------------------------
# STEP E: Vector Search
# -----------------------------------------------
def vector_search(query, index, model, chunks, top_k=10):
    """
    Search using meaning.
    Converts question to vector, finds closest chunk vectors.
    """
    # Convert question to vector
    query_vector = model.encode([query])

    # Find top_k closest vectors in FAISS
    distances, indexes = index.search(np.array(query_vector), top_k)

    results = []
    for i, idx in enumerate(indexes[0]):
        results.append({
            "chunk_text": chunks[idx]["chunk_text"],
            "title": chunks[idx]["title"],
            "authors": chunks[idx]["authors"],
            "year": chunks[idx]["year"],
            "url": chunks[idx]["url"],
            "score": float(distances[0][i]),
            "search_type": "vector"
        })
    return results


# -----------------------------------------------
# STEP F: Combine Both Results (Hybrid Search)
# -----------------------------------------------
def hybrid_search(query, bm25, index, model, chunks, top_k=10):
    """
    Run both searches and combine results.
    Why: BM25 finds exact keyword matches, vector finds meaning matches.
    Together they are much better than either alone.
    """
    # Get top 10 from each
    bm25_results = bm25_search(query, bm25, chunks, top_k)
    vector_results = vector_search(query, index, model, chunks, top_k)

    # Combine and remove duplicates
    # Use chunk_text as unique identifier
    seen = set()
    combined = []

    for result in bm25_results + vector_results:
        if result["chunk_text"] not in seen:
            seen.add(result["chunk_text"])
            combined.append(result)

    print(f"✅ Hybrid search found {len(combined)} unique results")
    return combined  # returns up to 20 unique results


# -----------------------------------------------
# MAIN — Test everything
# -----------------------------------------------
if __name__ == "__main__":
    # Load data
    chunks = load_chunks()

    # Build indexes
    bm25 = build_bm25_index(chunks)
    index, model, embeddings = build_vector_index(chunks)

    # Test with a sample question
    query = "How does attention work in transformers?"
    print(f"\n🔍 Searching for: '{query}'\n")

    results = hybrid_search(query, bm25, index, model, chunks)

    # Print top 5 results
    print("\n📄 Top Results:")
    for i, result in enumerate(results[:5]):
        print(f"\n--- Result {i+1} ---")
        print(f"Title: {result['title']}")
        print(f"Year: {result['year']}")
        print(f"Found by: {result['search_type']}")
        print(f"Text: {result['chunk_text'][:200]}...")