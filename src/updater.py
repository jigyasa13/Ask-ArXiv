import arxiv
import json
import os
import faiss
import numpy as np
import schedule
import time
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# -----------------------------------------------
# STEP A: Load existing chunks
# -----------------------------------------------
def load_existing_chunks():
    """
    Load whatever chunks we already have saved.
    Why: We need to add NEW chunks to existing ones,
    not replace everything from scratch.
    """
    chunks_path = "../data/chunks.json"

    if os.path.exists(chunks_path):
        with open(chunks_path, "r") as f:
            chunks = json.load(f)
        print(f"✅ Loaded {len(chunks)} existing chunks")
    else:
        chunks = []
        print("⚠️ No existing chunks found. Starting fresh.")

    return chunks


# -----------------------------------------------
# STEP B: Fetch only NEW papers from ArXiv
# -----------------------------------------------
def fetch_new_papers():
    """
    Fetch papers published in the last 24 hours only.

    Why only last 24 hours?
    - We run this script every day
    - We only want NEW papers, not ones we already have
    - Fetching everything every day would create duplicates
    """
    print(f"\n⏳ Fetching new papers from ArXiv...")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    client = arxiv.Client()

    SEARCH_QUERIES = [
        "transformer attention mechanism",
        "large language models",
        "retrieval augmented generation",
        "deep learning optimization",
        "neural network training"
    ]

    new_papers = []

    for query in SEARCH_QUERIES:
        search = arxiv.Search(
            query=query,
            max_results=5,  # only 5 per query per day — enough for daily updates
            sort_by=arxiv.SortCriterion.SubmittedDate  # newest first
        )

        for paper in client.results(search):
            # Only include papers from last 24 hours
            paper_date = paper.published.replace(tzinfo=None)
            yesterday = datetime.now() - timedelta(days=1)

            if paper_date >= yesterday:
                new_papers.append({
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors],
                    "year": paper.published.year,
                    "abstract": paper.summary,
                    "url": paper.entry_id
                })

    print(f"✅ Found {len(new_papers)} new papers from last 24 hours")
    return new_papers


# -----------------------------------------------
# STEP C: Chunk new papers
# -----------------------------------------------
def chunk_new_papers(new_papers, chunk_size=200):
    """
    Cut new papers into chunks — same as Week 2.
    Why: Consistency. All chunks in our system must be
    the same size so search works properly.
    """
    new_chunks = []

    for paper in new_papers:
        text = paper["abstract"]
        words = text.split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            new_chunks.append({
                "chunk_text": chunk,
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "url": paper["url"]
            })

    print(f"✅ Created {len(new_chunks)} new chunks")
    return new_chunks


# -----------------------------------------------
# STEP D: Remove duplicates
# -----------------------------------------------
def remove_duplicates(existing_chunks, new_chunks):
    """
    Before adding new chunks, check if they already exist.

    Why: If the same paper appears in multiple queries
    or was already added yesterday, we don't want duplicates.
    We use the paper URL as a unique identifier.
    """
    existing_urls = set(chunk["url"] for chunk in existing_chunks)

    unique_new_chunks = [
        chunk for chunk in new_chunks
        if chunk["url"] not in existing_urls
    ]

    print(f"✅ {len(unique_new_chunks)} unique new chunks after deduplication")
    return unique_new_chunks


# -----------------------------------------------
# STEP E: Save updated chunks
# -----------------------------------------------
def save_updated_chunks(all_chunks):
    """
    Save the combined old + new chunks back to JSON.
    Why: This is our master database of all chunks.
    Every search runs against this file.
    """
    with open("../data/chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"✅ Saved {len(all_chunks)} total chunks to data/chunks.json")


# -----------------------------------------------
# STEP F: Rebuild search indexes
# -----------------------------------------------
def rebuild_indexes(all_chunks):
    """
    After adding new chunks, rebuild both search indexes.

    Why rebuild instead of just adding?
    - BM25 needs to know about ALL chunks to score properly
    - FAISS needs all vectors in the right order
    - Rebuilding ensures everything is consistent

    In production at scale (millions of chunks) you would
    do incremental updates instead of full rebuilds.
    But for our project size, full rebuild is fine.
    """
    print("⏳ Rebuilding search indexes...")

    # Rebuild BM25
    tokenized = [c["chunk_text"].lower().split() for c in all_chunks]
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized)
    print("✅ BM25 index rebuilt")

    # Rebuild FAISS vector index using YOUR fine-tuned model
    model = SentenceTransformer("../models/cs-embedding-model")
    texts = [c["chunk_text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Save FAISS index to disk so other files can load it
    os.makedirs("../data", exist_ok=True)
    faiss.write_index(index, "../data/faiss_index.bin")

    print("✅ FAISS index rebuilt and saved")
    return bm25, index


# -----------------------------------------------
# STEP G: Main update function
# -----------------------------------------------
def run_daily_update():
    """
    This is the function that runs every day automatically.
    It ties all the steps together.
    """
    print("\n" + "="*50)
    print("🔄 DAILY UPDATE STARTED")
    print("="*50)

    # 1. Load what we have
    existing_chunks = load_existing_chunks()

    # 2. Fetch new papers
    new_papers = fetch_new_papers()

    if not new_papers:
        print("ℹ️ No new papers today. Skipping update.")
        return

    # 3. Chunk them
    new_chunks = chunk_new_papers(new_papers)

    # 4. Remove duplicates
    unique_new_chunks = remove_duplicates(existing_chunks, new_chunks)

    if not unique_new_chunks:
        print("ℹ️ All new papers already exist. Skipping update.")
        return

    # 5. Combine old + new
    all_chunks = existing_chunks + unique_new_chunks

    # 6. Save everything
    save_updated_chunks(all_chunks)

    # 7. Rebuild indexes
    rebuild_indexes(all_chunks)

    print("\n✅ DAILY UPDATE COMPLETE")
    print(f"   Total chunks now: {len(all_chunks)}")
    print(f"   New chunks added: {len(unique_new_chunks)}")
    print("="*50)


# -----------------------------------------------
# STEP H: Schedule to run every day
# -----------------------------------------------
def start_scheduler():
    """
    Schedule the update to run every day at 8:00 AM.

    Why 8:00 AM?
    - Most new ArXiv papers are submitted the night before
    - By morning they are processed and available
    - Users get fresh results at the start of their day

    How does schedule work?
    - It's a simple Python library
    - It keeps checking the time in a loop
    - When it matches "08:00", it runs our function
    """
    print("⏰ Scheduler started. Daily update will run at 08:00 AM.")
    print("   Press Ctrl+C to stop.\n")

    schedule.every().day.at("08:00").do(run_daily_update)

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # check every 60 seconds


# -----------------------------------------------
# MAIN
# -----------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting updater...\n")
    print("Choose mode:")
    print("1. Run update RIGHT NOW (for testing)")
    print("2. Start daily scheduler (runs every day at 8 AM)")

    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        run_daily_update()
    elif choice == "2":
        start_scheduler()
    else:
        print("Invalid choice. Running update now...")
        run_daily_update()