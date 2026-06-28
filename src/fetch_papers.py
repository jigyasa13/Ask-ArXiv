import arxiv
import os
import json

# -----------------------------------------------
# STEP A: Define what papers we want to download
# -----------------------------------------------
SEARCH_QUERIES = [
    "transformer attention mechanism",
    "large language models",
    "retrieval augmented generation",
    "deep learning optimization",
    "neural network training"
]

MAX_PAPERS = 100  # start with 100, you can increase later

# -----------------------------------------------
# STEP B: Download papers from ArXiv
# -----------------------------------------------
def fetch_papers():
    all_papers = []
    client = arxiv.Client()  # ← this is the fix

    for query in SEARCH_QUERIES:
        print(f"Fetching papers for: {query}")

        search = arxiv.Search(
            query=query,
            max_results=MAX_PAPERS // len(SEARCH_QUERIES),
            sort_by=arxiv.SortCriterion.Relevance
        )

        for paper in client.results(search):  # ← changed this line
            all_papers.append({
                "title": paper.title,
                "authors": [str(a) for a in paper.authors],
                "year": paper.published.year,
                "abstract": paper.summary,
                "url": paper.entry_id
            })

    print(f"\n✅ Total papers fetched: {len(all_papers)}")
    return all_papers


# -----------------------------------------------
# STEP C: Cut each paper into small chunks
# -----------------------------------------------
def chunk_papers(papers, chunk_size=200):
    """
    Cut each paper abstract into chunks of ~200 words.
    Why 200 words? Small enough for AI to process,
    big enough to contain useful information.
    """
    all_chunks = []

    for paper in papers:
        text = paper["abstract"]
        words = text.split()

        # Split into groups of chunk_size words
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])

            all_chunks.append({
                "chunk_text": chunk,
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "url": paper["url"]
            })

    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks


# -----------------------------------------------
# STEP D: Save everything to a file
# -----------------------------------------------
def save_chunks(chunks):
    """
    Save chunks as a JSON file in the data/ folder.
    Why JSON? Easy to read, easy to load later in other files.
    """
    os.makedirs("../data", exist_ok=True)  # create data/ folder if it doesn't exist

    with open("../data/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Chunks saved to data/chunks.json")


# -----------------------------------------------
# MAIN — Run everything
# -----------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting paper fetch...\n")
    papers = fetch_papers()
    chunks = chunk_papers(papers)
    save_chunks(chunks)
   