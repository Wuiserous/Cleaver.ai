import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import datetime

# --- CONFIGURATION ---
DB_DIR = "db_memory"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class MemoryManager:
    def __init__(self):
        print("🧠 Initializing Memory Manager (ChromaDB + MiniLM)...")
        # Initialize Embedding Model (Runs locally)
        self.encoder = SentenceTransformer(EMBED_MODEL_NAME)

        # Initialize Vector DB (Persistent)
        self.client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=True))

        # Collection 1: Conversation History (RAG)
        self.history_collection = self.client.get_or_create_collection(
            name="chat_history",
            metadata={"hnsw:space": "cosine"}
        )

        # Collection 2: Skills/Tools Library
        self.skills_collection = self.client.get_or_create_collection(
            name="skills_library",
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding(self, text):
        return self.encoder.encode(text).tolist()

    # --- HISTORY / CONTEXT ---
    def save_interaction(self, role, content):
        """Saves a single message to history."""
        if not isinstance(content, str):
            return  # Skip image data for text DB for now

        timestamp = datetime.datetime.now().isoformat()
        doc_id = f"{timestamp}_{role}"

        self.history_collection.add(
            documents=[content],
            metadatas=[{"role": role, "timestamp": timestamp}],
            ids=[doc_id],
            embeddings=[self._get_embedding(content)]
        )

    def retrieve_context(self, query, n_results=5):
        """Finds past conversations relevant to the current query."""
        results = self.history_collection.query(
            query_embeddings=[self._get_embedding(query)],
            n_results=n_results
        )

        context_str = ""
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                context_str += f"[{meta['timestamp']} - {meta['role']}]: {doc}\n"

        return context_str

    # --- SKILLS / TOOLS ---
    def save_skill(self, name, code, description, usage_example):
        """Saves a python script as a 'Skill'."""
        # We embed the description AND the usage example so semantic search finds it easily
        searchable_text = f"{name}: {description}. Example: {usage_example}"

        self.skills_collection.upsert(
            ids=[name],
            documents=[code],  # The actual code is the document
            metadatas=[{
                "name": name,
                "description": description,
                "usage": usage_example
            }],
            embeddings=[self._get_embedding(searchable_text)]
        )
        return f"Skill '{name}' saved to Vector DB."

    def retrieve_skill(self, query, n_results=1):
        """Finds the code for a skill based on a natural language request."""
        results = self.skills_collection.query(
            query_embeddings=[self._get_embedding(query)],
            n_results=n_results
        )

        if results['ids'] and results['ids'][0]:
            # Return the code and the metadata
            skill_name = results['ids'][0][0]
            code = results['documents'][0][0]
            desc = results['metadatas'][0][0]['description']
            return {"name": skill_name, "code": code, "description": desc}

        return None

    def list_all_skills(self):
        """Returns a list of all available skill names."""
        # Chroma doesn't have a cheap 'list all', so we peek
        count = self.skills_collection.count()
        if count == 0: return []
        data = self.skills_collection.get(limit=count, include=['metadatas'])
        return [m['name'] for m in data['metadatas']]


# Singleton Instance
memory = MemoryManager()


