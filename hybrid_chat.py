# hybrid_chat.py (Using Google Gemini API)
from typing import List, Tuple
from pinecone import Pinecone
from pinecone.exceptions import NotFoundException
from neo4j import GraphDatabase
import config
# --- NEW: Import and configure the Google Gemini library ---
import google.generativeai as genai

genai.configure(api_key=config.GOOGLE_API_KEY)

# --- Config ---
# --- NEW: Define Gemini models to use ---
EMBEDDING_MODEL = "models/text-embedding-004"
CHAT_MODEL_ID = "gemini-2.0-flash-lite"

TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# --- Initialize clients ---
pc = Pinecone(api_key=config.PINECONE_API_KEY)

embedding_cache = {}

def embed_text(text: str) -> List[float]:
    """Get embedding for a text string, using a cache."""
    # Check if the embedding is already in the cache
    if text in embedding_cache:
        return embedding_cache[text]
    
    # If not, call the API
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    embedding = resp.data[0].embedding
    
    # Store the result in the cache before returning
    embedding_cache[text] = embedding
    return embedding

# --- NEW: Initialize the Gemini Chat Model with a system instruction ---
# This is the modern way to set the persona for the model.
system_instruction = (
    "You are an expert travel assistant for Vietnam. Your goal is to create a helpful, coherent, and engaging travel itinerary or recommendation based on the user's request. "
    "Use the provided 'semantic search results' to understand the user's intent and find relevant starting points (like cities, hotels, or activities). "
    "Use the 'graph context' to find connections and related information (e.g., what activities are in a city, what city a hotel is in). "
    "Synthesize information from both sources to give a comprehensive answer. "
    "When you mention a specific place, activity, or hotel, please cite its ID in parentheses, like (id: city_hanoi)."
)
chat_model = genai.GenerativeModel(
    model_name=CHAT_MODEL_ID,
    system_instruction=system_instruction
)

# --- Connect to Pinecone index ---
try:
    pc.describe_index(INDEX_NAME)
except NotFoundException:
    print(f"Index '{INDEX_NAME}' not found. Please run the Gemini version of pinecone_upload.py first.")
    exit()
index = pc.Index(INDEX_NAME)

# --- Connect to Neo4j ---
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)
driver.verify_connectivity()

# --- Helper functions ---
# --- NEW: embed_text now uses the Gemini API ---
def embed_text(text: str) -> List[float]:
    """Get embedding for a text string using the Gemini API."""
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    return result['embedding']

def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using a Gemini embedding."""
    vec = embed_text(query_text)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True, include_values=False)
    print(f"DEBUG: Pinecone found {len(res.matches)} results.")
    return res.matches

# This function remains the same as it only interacts with Neo4j.
def fetch_graph_context(node_ids: List[str]):
    if not node_ids: return []
    query = """
    UNWIND $node_ids AS nid
    MATCH (n:Entity {id: nid})-[r]-(m:Entity)
    RETURN n.id AS source_id, type(r) AS rel, m.id AS target_id, m.name AS target_name,
           m.description AS target_desc, labels(m) AS target_labels
    LIMIT 15
    """
    with driver.session() as session:
        result = session.run(query, node_ids=node_ids)
        facts = [record.data() for record in result]
    print(f"DEBUG: Graph query returned {len(facts)} facts.")
    return facts

# --- NEW: build_prompt now just creates the user-facing part of the prompt ---
def build_user_prompt(user_query: str, pinecone_matches, graph_facts) -> str:
    """Builds the user-facing part of the prompt for the Gemini model."""
    vec_context_str = "\n".join([
        f"- (id: {m.id}), name: {m.metadata.get('name', 'N/A')}, type: {m.metadata.get('type', 'N/A')}, city: {m.metadata.get('city', 'N/A')}, semantic score: {m.score:.2f}"
        for m in pinecone_matches
    ])

    graph_context_str = "\n".join([
        f"- ({f['source_id']}) is [{f['rel']}] ({f['target_id']}) {f['target_name']}"
        for f in graph_facts
    ])

    return (
        f"User's travel question: '{user_query}'\n\n"
        "Here is some context to help you answer:\n\n"
        "--- Semantic Search Results ---\n"
        f"{vec_context_str}\n\n"
        "--- Graph Context (Relationships) ---\n"
        f"{graph_context_str}\n\n"
        "--- Your Task ---\n"
        "Based on all the information above, please provide a clear, step-by-step itinerary or a set of recommendations that directly answers the user's question. Be creative and practical."
    )

# --- NEW: call_chat now uses the Gemini API ---
def call_chat(user_prompt: str):
    """Calls the Gemini Chat Model."""
    response = chat_model.generate_content(user_prompt)
    return response.text

# --- Interactive chat loop ---
def main():
    print("Welcome to the Hybrid AI Travel Assistant (Google Gemini) Type 'exit' to quit.")
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # 1. Vector Search (with Gemini embedding)
        matches = pinecone_query(query, top_k=TOP_K)
        
        # 2. Graph Search
        match_ids = [m.id for m in matches]
        graph_facts = fetch_graph_context(match_ids)
        
        # 3. Build Prompt
        user_prompt = build_user_prompt(query, matches, graph_facts)
        
        # 4. Generate Answer (with Gemini chat model)
        answer = call_chat(user_prompt)
        
        print("\n=== Assistant's Recommendation ===\n")
        print(answer)
        print("\n==================================\n")

if __name__ == "__main__":
    main()