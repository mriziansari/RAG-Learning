import os
from google.genai import Client
from google.genai.types import GenerateContentConfig
import dotenv
from retriever import Retriever, Reranker, QueryEmbedder

dotenv.load_dotenv()

class ResponseGenerator:
    def __init__(self):
        self.gclient = Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
    def generate_response(self, query_text: str, retrieved_chunks: list) -> str:
        """Generates a response using the LLM and retrieved context."""
        
        if not retrieved_chunks:
            return "No relevant context found to answer the query."

        context_text = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        
        SYSTEM_PROMPT = """You are an Expert Knowledge Assistant. Your goal is to provide accurate, helpful answers based STRICTLY on the provided context.

### INSTRUCTIONS:
1. **Analyze** the user's query and the provided context.
2. **Synthesize** an answer using ONLY the information found in the context tags.
3. **Guardrails**: If the answer is not explicitly stated in the context, strictly reply: "I cannot find the answer to this question in the provided context." Do not hallucinate or use outside knowledge.
4. **Tone**: Professional, concise, and direct.

### FORMATTING:
- Use bullet points if you need to list multiple points.
- Be clear and structured.
"""

        USER_PROMPT = f"""
<context>
{context_text}
</context>
<query>
{query_text}
</query>
"""
        config = GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        )
        try:
            response = self.gclient.models.generate_content(
                model="gemini-2.5-flash",
                config=config,
                contents=USER_PROMPT,
            )
            return response.text
        except Exception as e:
            return f"Error generating content: {e}"

def main():
    # query_text = "By how many tons per capita must the urban carbon footprint drop to meet the 2035 target from the 2024 baseline?"
    # query_text = "How does the 'Right to Shade' regulation in Phoenix potentially impact the temperature of neighboring buildings?"
    # query_text = "Identify the transit mode that is the most energy-efficient according to Table 1 and explain its fire rating if mentioned?"
    # query_text = "The report mentions that Autonomous Vehicles eliminate traffic jams. Is this consistent with the findings in San Francisco?"
    query_text = "What is the specific initial cost barrier mentioned for Kinetic Pavements, and which city conducted the pilot study?"
    
    # 1. Retrieve the best section (Level 1)
    query_embedder = QueryEmbedder()
    retriever = Retriever(query_embedder)
    reranker = Reranker(query_embedder)
    relevant_section_id = retriever.recursive_retrieve(query_text,n_results=5)
    
    # 2. Retrieve all chunks from the best section (Level 2)
    retrieved_chunks = retriever.retrieve_context(query_text, section_id=relevant_section_id, n_results=5)
    
    # 3. Stage 1: Rerank retrieved chunks
    reranked_chunks = reranker.stage1_reranking(query_text, retrieved_chunks)
    # 4. Stage 2: Rerank retrieved chunks
    reranked_chunks = reranker.stage2_reranking(query_text, reranked_chunks)
    # 5. Generate Response
    response_generator = ResponseGenerator()
    response = response_generator.generate_response(query_text, reranked_chunks)
    
    print("-" * 100)
    print("Stage 1 Reranked Chunks:")
    for chunk in reranked_chunks:
        print("*" * 100)
        print("stage1_score: ",chunk["final_score"])
        print("cross_encoder_score: ",chunk["cross_encoder_score"])
        print("text: ",chunk["text"][:120])
    print("-" * 100)
    print("Response:")
    print(response)
    print("-" * 100)
        
if __name__ == "__main__":
    main()