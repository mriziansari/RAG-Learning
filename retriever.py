import chromadb
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/urban_infrastructure")
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def recursive_retrieve(self, query_text: str, n_results: int = 5) -> list:
        """ 
        Retrieve the most relevant Section (Level 1) from ChromaDB.
        Returns the 'section' ID (e.g. 'section_5').
        """
        try:
            collection = self.client.get_collection("router_collection")
            
            # Generate embedding for the query
            query_embedding = self.model.encode([query_text]).tolist()

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
            query_embedding = self.model.encode([query_text]).tolist()

            filters = [{"level": 2}]
            if section_id and len(section_id) > 0:
                filters.append({"parent_section": {"$in": section_id}})

            result = collection.query(
                query_embeddings=query_embedding,
                where={
                    "$and": filters
                },
                n_results=n_results
            )

            print(f"Retrieved {len(result['documents'][0])} chunks")
            for i, doc in enumerate(result['documents'][0]):
                print("-" * 100)
                print(f"chunk_{i}")
                print(doc)
            
            if result["documents"] and result["documents"][0]:
                return result["documents"][0]
            return []
            
        except Exception as e:
            print(f"Error accessing ChromaDB (Level 2): {e}")
            return []
        

def main():
    retriever = Retriever()
    
    # query_text = "By how many tons per capita must the urban carbon footprint drop to meet the 2035 target from the 2024 baseline?"
    # query_text = "How does the 'Right to Shade' regulation in Phoenix potentially impact the temperature of neighboring buildings?"
    query_text = "Identify the transit mode that is the most energy-efficient according to Table 1 and explain its fire rating if mentioned?"
    # query_text = "The report mentions that Autonomous Vehicles eliminate traffic jams. Is this consistent with the findings in San Francisco?"
    # query_text = "What is the specific initial cost barrier mentioned for Kinetic Pavements, and which city conducted the pilot study?"
    
    # 1. Retrieve the best section (Level 1)
    relevant_section_id = retriever.recursive_retrieve(query_text, n_results=5)
    
    if relevant_section_id:
        # 2. Retrieve specific chunks within that section (Level 2)
        retrieved_chunks = retriever.retrieve_context(query_text, section_id=relevant_section_id, n_results=5)
    else:
        # Fallback: Searching all Level 2 chunks if no section found (optional, but good for robustness)
        print("No specific section matched. Searching all details...")
        retrieved_chunks = retriever.retrieve_context(query_text, section_id=None)

if __name__ == "__main__":
    main()