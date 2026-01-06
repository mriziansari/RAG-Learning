import uuid
import time
import chromadb


class MemoryStore:
    def __init__(self, embedding_fn, collection_name="agent_memory"):
        self.embedding_fn = embedding_fn
        self.client = chromadb.PersistentClient(path="data/agent_memory")
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(self, content: str, memory_type: str = "semantic", confidence: float = 0.80, metadata: dict | None = None):
        """Store summrized long term memory"""
        print("\nI am in the memory store Add function")
        memory_id = str(uuid.uuid4())
        embedding = self.embedding_fn(content)

        meta = {
            "memory_type": memory_type,
            "confidence": confidence,
            "created_at": time.time(),
            "last_accessed": time.time(),
        }
        print("\n >>>>>>>>>content to add in the memory:<<<<<<<<<<", [content])
        print("=" * 100)
        if metadata:
            meta.update(metadata)
        print("Memory added metadata:", meta)
        self.collection.add(
            ids=[memory_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def retrieve(self, query: str, top_k: int = 3,min_score: float = 0.65):
        """Retrieve relevant memories based on query"""

        embedding = self.embedding_fn(query)

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "documents","distances"]
        )
        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1 - dist  # cosine distance → similarity
            print("\n<<<<<<<Confidence score:>>>>>>>\n", score)
            if score >= min_score:
                memories.append({
                    "content": doc,
                    "score": score,
                    "metadata": meta
                })

        return memories
