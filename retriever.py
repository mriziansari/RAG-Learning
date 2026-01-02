import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import re   
from query_expander import QueryExpander

class QueryEmbedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, query_text: str) -> np.ndarray:
        return self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True
            )[0]

class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/urban_infrastructure")
        self.query_embedder = QueryEmbedder()

    def recursive_retrieve(self, query_text: list, n_results: int = 5) -> list:
        """ 
        Retrieve the most relevant Section (Level 1) from ChromaDB.
        Returns the 'section' ID (e.g. 'section_5').
        """
        try:
            collection = self.client.get_collection("router_collection")
            
            # Generate embedding for the query
            query_embedding = self.query_embedder.embed_query(query_text)

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
                return section_ids
            return []
            
        except Exception as e:
            print(f"Error accessing ChromaDB (Level 1): {e}")
            return []

    def retrieve_context(self, query_text: str, section_id=None, n_results: int = 5) -> list:
        """
        Retrieve relevant Details (Level 2) from ChromaDB.
        If section_id is provided, filter by that parent_section.
        """
        if section_id is None:
            section_id = []
        try:
            collection = self.client.get_collection("answer_collection")
            
            # Generate embedding for the query
            query_embedding = self.query_embedder.embed_query(query_text)

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

            # print(f"Retrieved {len(result['documents'][0])} chunks")
            # for i, doc in enumerate(result['documents'][0]):
            #     print("-" * 100)
            #     print(f"chunk_{i}")
            #     print(doc)
            
            if result["documents"] and result["documents"][0]:
                return retrived_chunks
            return []
            
        except Exception as e:
            print(f"Error accessing ChromaDB (Level 2): {e}")
            return []
        
    def retrive_chunks(self, extended_queries: list):
        """
        Retrieve chunks for each query and merge them.
        args:
            extended_queries: list of extended queries retrived from expand_query function in query_expander.py
        """
        all_chunks = []
        for query in extended_queries:
           section_ids = self.recursive_retrieve(query)
           chunks = self.retrieve_context(query, section_ids)
           for c in chunks:
               c["source_query"] = query
           all_chunks.extend(chunks)
        
        merged_chunks = self.deduplicate(all_chunks)
        return merged_chunks

    def deduplicate(self, chunks):
        """
        Deduplicate chunks based on chunk_id and distance.
        args:
            chunks: list of chunks retrived from retrieve_context function
        """
        seen = {}
        for i, c in enumerate(chunks):
            key = f"{c['metadata']['doc_id']}_{c['metadata']['chunk_index']}"
            if key not in seen or c["distance"] < seen[key]["distance"]:
                seen[key] = c
        return list(seen.values())
        
        
class Reranker:
    def __init__(self, model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.query_embedder = QueryEmbedder()

        # this model will be used for stage two reranking. It is slow but more accurate so we use it for top_n chunks from stage 1
        self.cross_encoder = CrossEncoder(model_name)

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

    def stage1_reranking(self, query_text: str, retrieved_chunks: list, top_n: int = 5) -> list:
        """
        Stage 1: Rerank retrieved chunks for stage 1.
        """
        scored_chunks = []
        query_embedding = self.query_embedder.embed_query(query_text)
        for chunk in retrieved_chunks:
            chunk_embedding = np.array(chunk["embedding"])
            sim = self.cosine_similarity(query_embedding, chunk_embedding)
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
        
    def stage2_reranking(self, query_text, stage1_reranked_chunks, top_n: int = 3):
        """
        Stage 2: Rerank retrieved chunks for stage 2.
        args:
            query_text: The query text.
            stage1_reranked_chunks: The chunks reranked by stage 1.
            top_n: The number of chunks to return.
        """

        pairs = [
            (query_text, chunk["text"]) 
            for chunk in stage1_reranked_chunks
            ]

        scores = self.cross_encoder.predict(pairs)
        
        for i, score in enumerate(scores):
            stage1_reranked_chunks[i]["cross_encoder_score"] = float(score)
        
        stage1_reranked_chunks.sort(
            key=lambda x: x["cross_encoder_score"],
            reverse=True
            )
        return stage1_reranked_chunks[:top_n]
        