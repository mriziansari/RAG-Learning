import json
from google.genai import Client
from google.genai.types import GenerateContentConfig
import dotenv
from retriever import Retriever, Reranker, QueryEmbedder

dotenv.load_dotenv()

class ResponseGenerator:
    def __init__(self, client: Client, model_name: str):
        self.gclient = client
        self.model = model_name
        
    def generate_response(self, query_text: str, retrieved_context: list, source: str = "rag") -> dict:
        """Generates a response using the LLM and retrieved context."""
        
        if source == "memory":
           context_text = retrieved_context
           SYSTEM_PROMPT = """You are an Expert Knowledge Assistant. Your goal is to provide accurate,
        helpful answers based STRICTLY on the provided context which is retrieved from memory.
        Return STRICT JSON:
        {{
          "answer": "...",
          "memory_summary": "..."
        }} """

        else:
            context_text = "\n\n".join([chunk["text"] for chunk in retrieved_context])
            SYSTEM_PROMPT = """You are an Expert Knowledge Assistant. Your goal is to provide accurate, helpful answers based STRICTLY on the provided context.
        Also generate a short reasonable memory summary to save in the memory store. Summery must include the query and the answer summarized context.

### INSTRUCTIONS:
1. **Analyze** the user's query and the provided context.
2. **Synthesize** an answer using ONLY the information found in the context tags.
3. **Guardrails**: If the answer is not explicitly stated in the context, strictly reply: "I cannot find the answer to this question in the provided context." Do not hallucinate or use outside knowledge.
4. **Tone**: Professional, concise, and direct.

Return STRICT JSON:
{{
  "answer": "...",
  "memory_summary": "..."
}}
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
            response_mime_type="application/json",
        )
        try:
            response = self.gclient.models.generate_content(
                model=self.model,
                config=config,
                contents=USER_PROMPT,
            )
            # Parse the JSON response
            try:
                parsed_response = json.loads(response.text)
                print("Response generated:", parsed_response)
                return parsed_response
            except json.JSONDecodeError:
                print("Failed to parse JSON response. Raw text:", response.text)
                return {"answer": response.text, "memory_summary": ""}
                
        except Exception as e:
            return {"answer": f"Error generating content: {e}", "memory_summary": ""}

