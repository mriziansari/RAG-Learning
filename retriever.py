import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import re   

class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/urban_infrastructure")
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def query_embedding(self, query_text: str) -> list:
        # Generate embedding for the query
        return self.model.encode([query_text]).tolist()

    def recursive_retrieve(self, query_text: str, n_results: int = 5) -> list:
        """ 
        Retrieve the most relevant Section (Level 1) from ChromaDB.
        Returns the 'section' ID (e.g. 'section_5').
        """
        try:
            collection = self.client.get_collection("router_collection")
            
            # Generate embedding for the query
            query_embedding = self.query_embedding(query_text)

            # Query for Level 1 chunks (Sections)
            result = collection.query(
                query_embeddings=query_embedding,
                where={
                    "$and": [
                        {"level": 1},
                        {"role": "router"}
                    ]
                },
                n_results=n_results
            )

            # Parse the result to get the section ID
            # result['metadatas'] is a list of lists (one list per query string)
            
            if result["metadatas"] and result["metadatas"][0]:
                section_ids = [
                    metadata.get("section_id")
                    for metadata in result["metadatas"][0]
                ]
                print(f"Identify most relevant section: {', '.join(section_ids)}")
                return section_ids
            return []
            
        except Exception as e:
            print(f"Error accessing ChromaDB (Level 1): {e}")
            return []

    def retrieve_context(self, query_text: str, section_id: list = [], n_results: int = 5) -> list:
        """
        Retrieve relevant Details (Level 2) from ChromaDB.
        If section_id is provided, filter by that parent_section.
        """
        try:
            collection = self.client.get_collection("answer_collection")
            
            # Generate embedding for the query
            query_embedding = self.query_embedding(query_text)

            filters = [{"level": 2}]
            if section_id and len(section_id) > 0:
                filters.append({"parent_section": {"$in": section_id}})

            result = collection.query(
                query_embeddings=query_embedding,
                where={
                    "$and": filters
                },
                n_results=n_results,
                include=["metadatas", "documents", "embeddings", "distances"]
            )

            retrived_chunks = []
            for i in range(len(result["documents"][0])):
                retrived_chunks.append(
                    {
                        "text": result["documents"][0][i],
                        "metadata": result["metadatas"][0][i],
                        "embedding": result["embeddings"][0][i],
                        "distance": result["distances"][0][i]
                    }
                )

            print(f"Retrieved {len(result['documents'][0])} chunks")
            for i, doc in enumerate(result['documents'][0]):
                print("-" * 100)
                print(f"chunk_{i}")
                print(doc)
            
            if result["documents"] and result["documents"][0]:
                return retrived_chunks
            return []
            
        except Exception as e:
            print(f"Error accessing ChromaDB (Level 2): {e}")
            return []
        
   
class Reranker:
    def __init__(self):
        pass
    def cosine_similarity(self, a, b):
        # Compute cosine similarity between two vectors.
        # Dot product divided by product of norms returns similarity score between -1 and 1.
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def tokenize(self, text):
        # Use regex to extract lowercase words ignoring punctuation.
        # This creates a set of tokens for overlap comparison.
        return set(re.findall(r"\b\w+\b", text.lower()))

    def keyword_overlap_score(self, query, text):
        # Tokenize both query and chunk text
        q_tokens = self.tokenize(query)
        t_tokens = self.tokenize(text)

        # If there are no query tokens, return 0
        if not q_tokens:
            return 0.0

        # Compute intersection size divided by number of query tokens
        return len(q_tokens & t_tokens) / len(q_tokens)
    
    def chunk_quality_score(self, text):
        # Evaluate chunk length as heuristic for quality
        length = len(text)
        # Too short: likely incomplete context → low score
        if length < 200:
            return 0.3
        # Too long: could dilute relevance → moderate score
        if length > 700:
            return 0.6
        # In sweet spot → full score
        return 1.0

    def stage1_reranking(self, query_text: str, query_embedding: list, retrieved_chunks: list, top_n: int = 5) -> list:
        scored_chunks = []
        for chunk in retrieved_chunks:
            sim = self.cosine_similarity(query_embedding, chunk["embedding"])
            overlap = self.keyword_overlap_score(query_text, chunk["text"])
            quality = self.chunk_quality_score(chunk["text"])
            
            # Weighted composite score with tuned weights
            final_score = 0.65 * sim + 0.25 * overlap + 0.10 * quality
            
            scored_chunks.append({
                "final_score": final_score,
                "semantic": sim,
                "overlap": overlap,
                "quality": quality,
                "chunk": chunk,
                "distance": chunk["distance"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            })
        
        scored_chunks.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_chunks[:top_n]    
        

def main():
    retriever = Retriever()
    reranker = Reranker()
    
    # query_text = "By how many tons per capita must the urban carbon footprint drop to meet the 2035 target from the 2024 baseline?"
    # query_text = "How does the 'Right to Shade' regulation in Phoenix potentially impact the temperature of neighboring buildings?"
    # query_text = "Identify the transit mode that is the most energy-efficient according to Table 1 and explain its fire rating if mentioned?"
    query_text = "The report mentions that Autonomous Vehicles eliminate traffic jams. Is this consistent with the findings in San Francisco?"
    # query_text = "What is the specific initial cost barrier mentioned for Kinetic Pavements, and which city conducted the pilot study?"
    
    # 1. Retrieve the best section (Level 1)
    relevant_section_id = retriever.recursive_retrieve(query_text, n_results=5)
    
    if relevant_section_id:
        # 2. Retrieve specific chunks within that section (Level 2)
        retrieved_context = retriever.retrieve_context(query_text, section_id=relevant_section_id, n_results=5)
    else:
        # Fallback: Searching all Level 2 chunks if no section found (optional, but good for robustness)
        print("No specific section matched. Searching all details...")
        retrieved_context = retriever.retrieve_context(query_text, section_id=None)

    if retrieved_context:
        # 3. Stage 1: Rerank retrieved chunks
        reranked_chunks = reranker.stage1_reranking(query_text, retriever.query_embedding(query_text), retrieved_context)
        
        # 4. Print results
        for i, chunk in enumerate(reranked_chunks):
            print(f"\nChunk {i+1} (Score: {chunk['final_score']})")
            print(chunk['text'])
            print("-" * 100)

if __name__ == "__main__":
    main()