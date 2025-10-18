# pinecone_upload.py (Using Google Gemini API)
import json
import time
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
import config
# --- NEW: Import and configure the Google Gemini library ---
import google.generativeai as genai

genai.configure(api_key=config.GOOGLE_API_KEY)

# --- Config ---
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32 # Gemini API can handle larger batches, but 32 is safe
INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM # This is now 768 from config.py

# --- Initialize Pinecone client ---
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# --- NEW: Define the Gemini embedding model ---
EMBEDDING_MODEL = "models/text-embedding-004"

# --- Delete and Recreate Index (IMPORTANT because dimension changed) ---
if INDEX_NAME in pc.list_indexes().names():
    print(f"Deleting existing index '{INDEX_NAME}' to change vector dimensions...")
    pc.delete_index(INDEX_NAME)
    # It can take a few moments for the index to be fully deleted.
    print("Waiting for index deletion...")
    time.sleep(10)

print(f"Creating new serverless index: {INDEX_NAME} with dimension {VECTOR_DIM}")
pc.create_index(
    name=INDEX_NAME,
    dimension=VECTOR_DIM,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Connect to the index
index = pc.Index(INDEX_NAME)

# --- Helper functions ---
# --- NEW: get_embeddings function now uses the Gemini API ---
def get_embeddings(texts):
    """Generate embeddings using the Google Gemini API."""
    # The API takes a list of strings and returns embeddings for each.
    result = genai.embed_content(model=EMBEDDING_MODEL, content=texts)
    return result['embedding']

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# --- Main upload logic (no changes in this function's structure) ---
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone using Gemini embeddings...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]
        index.upsert(vectors)
        time.sleep(0.2) # Small delay to respect API rate limits

    print("All items uploaded successfully using Gemini embeddings.")
    print("You can view your index in the Pinecone dashboard.")
    print(index.describe_index_stats())

if __name__ == "__main__":
    main()