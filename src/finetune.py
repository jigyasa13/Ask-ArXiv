import json
import random
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

# -----------------------------------------------
# STEP A: Load our chunks
# -----------------------------------------------
def load_chunks():
    with open("../data/chunks.json", "r") as f:
        chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks")
    return chunks


# -----------------------------------------------
# STEP B: Create training data
# -----------------------------------------------
def create_training_data(chunks):
    """
    We create two types of pairs:

    POSITIVE pairs (similar):
    - A question generated from a chunk + that same chunk
    - These should have HIGH similarity score

    NEGATIVE pairs (not similar):
    - A question generated from one chunk + a random different chunk
    - These should have LOW similarity score

    Why do we need both?
    The model learns by contrast. It needs to see what IS
    relevant and what is NOT relevant at the same time.

    Why generate questions from chunks?
    Because we don't have real user questions with known answers.
    So we create synthetic (fake but realistic) questions
    from the paper content itself.
    """
    print("⏳ Creating training data...")
    training_examples = []

    for i, chunk in enumerate(chunks):
        text = chunk["chunk_text"]
        title = chunk["title"]

        # --- POSITIVE PAIR ---
        # Create a simple question from the chunk title
        # Why title? It summarizes what the chunk is about
        question = f"What is {title} about?"

        # Positive example: question is similar to its own chunk
        training_examples.append(
            InputExample(
                texts=[question, text],
                label=1.0  # 1.0 = very similar
            )
        )

        # --- NEGATIVE PAIR ---
        # Pick a random different chunk
        random_idx = random.randint(0, len(chunks) - 1)
        while random_idx == i:  # make sure it's actually different
            random_idx = random.randint(0, len(chunks) - 1)

        random_chunk = chunks[random_idx]["chunk_text"]

        # Negative example: question is NOT similar to random chunk
        training_examples.append(
            InputExample(
                texts=[question, random_chunk],
                label=0.0  # 0.0 = not similar
            )
        )

    print(f"✅ Created {len(training_examples)} training examples")
    print(f"   ({len(training_examples)//2} positive + {len(training_examples)//2} negative pairs)")
    return training_examples


# -----------------------------------------------
# STEP C: Fine-tune the model
# -----------------------------------------------
def finetune_model(training_examples):
    """
    We start with the pre-trained model from Week 3
    and continue training it on our CS paper data.

    Why start from pre-trained instead of scratch?
    - Training from scratch needs millions of examples
    - Fine-tuning needs only hundreds/thousands
    - The pre-trained model already understands English
    - We just teach it CS research language on top

    Why CosineSimilarityLoss?
    - Our labels are 0.0 (not similar) and 1.0 (similar)
    - This loss function pushes similar pairs closer together
      and different pairs further apart in vector space
    - Perfect for our use case

    Why batch_size=16?
    - Small enough to fit in laptop memory
    - Large enough to train efficiently

    Why 3 epochs?
    - 1 epoch = model sees all training data once
    - 3 epochs = sees it 3 times = learns better
    - More than 5 epochs risks overfitting (memorizing instead of learning)
    """
    print("\n⏳ Loading base model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create DataLoader — feeds data to model in batches
    train_dataloader = DataLoader(
        training_examples,
        shuffle=True,       # shuffle so model doesn't learn order
        batch_size=16
    )

    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model)

    print("🚀 Starting fine-tuning...")
    print("   This will take 10-20 minutes on a laptop.\n")

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,   # slowly increase learning rate at start
        show_progress_bar=True
    )

    # Save your fine-tuned model
    os.makedirs("../models", exist_ok=True)
    model.save("../models/cs-embedding-model")

    print("\n✅ Model fine-tuned and saved to models/cs-embedding-model")
    return model


# -----------------------------------------------
# STEP D: Quick test — compare old vs new model
# -----------------------------------------------
def compare_models(chunks):
    """
    Quick sanity check — does our model give better results
    than the generic one on a CS question?
    """
    from sentence_transformers import util

    query = "How does attention mechanism work in neural networks?"
    test_chunks = chunks[:50]  # test on first 50 chunks
    texts = [c["chunk_text"] for c in test_chunks]

    print("\n🔍 Comparing OLD model vs YOUR fine-tuned model...")
    print(f"Query: '{query}'\n")

    # Old model
    old_model = SentenceTransformer("all-MiniLM-L6-v2")
    old_embeddings = old_model.encode(texts)
    old_query_emb = old_model.encode([query])
    old_scores = util.cos_sim(old_query_emb, old_embeddings)[0]
    old_top = old_scores.topk(3)

    print("📌 OLD MODEL top 3 results:")
    for score, idx in zip(old_top.values, old_top.indices):
        print(f"  Score: {score:.4f} | {test_chunks[idx]['title']}")

    # Your fine-tuned model
    new_model = SentenceTransformer("../models/cs-embedding-model")
    new_embeddings = new_model.encode(texts)
    new_query_emb = new_model.encode([query])
    new_scores = util.cos_sim(new_query_emb, new_embeddings)[0]
    new_top = new_scores.topk(3)

    print("\n🚀 YOUR FINE-TUNED MODEL top 3 results:")
    for score, idx in zip(new_top.values, new_top.indices):
        print(f"  Score: {score:.4f} | {test_chunks[idx]['title']}")


# -----------------------------------------------
# MAIN
# -----------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting fine-tuning pipeline...\n")

    # Load chunks
    chunks = load_chunks()

    # Create training data
    training_examples = create_training_data(chunks)

    # Fine-tune model
    model = finetune_model(training_examples)

    # Compare old vs new
    compare_models(chunks)

    print("\n🎉 Done! Your custom model is saved in models/cs-embedding-model")