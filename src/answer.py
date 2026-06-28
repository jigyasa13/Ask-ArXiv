from dotenv import load_dotenv
import os
from reranker import load_reranker, rerank_results
from search import load_chunks, build_bm25_index, build_vector_index, hybrid_search

# -----------------------------------------------
# STEP A: Load API key and OpenAI client
# -----------------------------------------------
from groq import Groq
load_dotenv("../.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------------------------
# STEP B: Format chunks into readable context
# -----------------------------------------------
def format_context(top_3_chunks):
    """
    Convert our list of chunks into a clean text block
    that GPT can read easily.

    Why do we format it like this?
    - GPT needs to know which text came from which paper
    - We label each chunk clearly: [1], [2], [3]
    - Later GPT will cite using these numbers
    """
    context = ""
    for i, chunk in enumerate(top_3_chunks):
        context += f"""
[{i+1}] Title: {chunk['title']}
Authors: {', '.join(chunk['authors'][:3])}
Year: {chunk['year']}
URL: {chunk['url']}
Content: {chunk['chunk_text']}

"""
    return context


# -----------------------------------------------
# STEP C: Build the prompt
# -----------------------------------------------
def build_prompt(query, context):
    """
    This is the most important part of Week 5.

    We give GPT very strict instructions:
    1. Only use the context provided — not its own memory
    2. Cite every fact using [1], [2], [3]
    3. If the answer is not in the context, say so honestly
    4. Keep the answer clear and structured

    Why such strict instructions?
    - Without them GPT will mix its own knowledge in
    - We want answers ONLY from our research papers
    - Citations make the answer trustworthy and verifiable
    """
    prompt = f"""You are a research assistant that answers questions using ONLY the provided research paper excerpts below.

STRICT RULES:
1. Only use information from the context provided below. Do NOT use your own knowledge.
2. After every fact or sentence, cite the source like this: [1], [2], or [3]
3. If the context does not contain enough information to answer, say: "The provided papers do not contain enough information to answer this question."
4. Keep your answer clear, structured, and under 200 words.
5. At the end, list the full references used.

CONTEXT FROM RESEARCH PAPERS:
{context}

QUESTION: {query}

ANSWER (with citations):"""
    return prompt


# -----------------------------------------------
# STEP D: Call GPT and get the answer
# -----------------------------------------------
def generate_answer(query, top_3_chunks):
    """
    Send the question + context to GPT and get a cited answer back.

    Why gpt-4o-mini?
    - Cheaper than gpt-4o but still very capable
    - Fast response time
    - Good at following strict instructions like citation rules

    Why temperature=0?
    - Temperature controls randomness
    - 0 means GPT gives the most factual, consistent answer
    - We don't want creativity here — we want accuracy
    """
    context = format_context(top_3_chunks)
    prompt = build_prompt(query, context)

    print("⏳ Generating answer...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a precise research assistant. Always cite sources. Never make things up."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,      # no randomness — we want facts
        max_tokens=500      # keep answers concise
    )

    answer = response.choices[0].message.content
    return answer


# -----------------------------------------------
# STEP E: Pretty print the final answer
# -----------------------------------------------
def print_answer(query, answer, top_3_chunks):
    print("\n" + "="*60)
    print(f"❓ QUESTION: {query}")
    print("="*60)
    print(f"\n💬 ANSWER:\n{answer}")
    print("\n" + "="*60)
    print("📚 SOURCES USED:")
    for i, chunk in enumerate(top_3_chunks):
        print(f"[{i+1}] {chunk['title']} ({chunk['year']}) — {chunk['url']}")
    print("="*60)


# -----------------------------------------------
# MAIN — Full pipeline test
# -----------------------------------------------
if __name__ == "__main__":
    print("🚀 Loading everything...\n")

    # Load data and build indexes (from Week 2 & 3)
    chunks = load_chunks()
    bm25 = build_bm25_index(chunks)
    index, model, embeddings = build_vector_index(chunks)

    # Load reranker (from Week 4)
    reranker = load_reranker()

    # Test question
    query = "How does the attention mechanism work in transformers?"

    # Step 1: Hybrid search → 20 results
    results = hybrid_search(query, bm25, index, model, chunks)

    # Step 2: Rerank → top 3
    top_3 = rerank_results(query, results, reranker)

    # Step 3: Generate answer with citations
    answer = generate_answer(query, top_3)

    # Print everything nicely
    print_answer(query, answer, top_3)