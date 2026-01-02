import os
from google.genai import Client
from google.genai.types import GenerateContentConfig
import dotenv
from retriever import Retriever, Reranker, QueryEmbedder

dotenv.load_dotenv()

class ResponseGenerator:
    def __init__(self, client: Client):
        self.gclient = client
        
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

