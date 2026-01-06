import json
from google.genai.types import GenerateContentConfig



class ReasoningPlanner:
    """
    Decides the NEXT action for an Agentic RAG system
    based on the evolving execution state.
    """

    def __init__(self, llm_client, model_name: str):
        self.llm_client = llm_client
        self.model = model_name

    def next_action(self, state: dict) -> dict:
        """
        Decide the next tool to call based on current state.
        The planner NEVER mutates state. It only reads it.
        """
        print("I am in the reasoning planner")
        # --- Safely derive state values ---
        user_query = state.get("user_query", "")
        retrieved_chunks = state.get("retrieved_chunks") or []
        reranked_chunks = state.get("reranked_chunks") or []
        observations = state.get("observations") or []
        memories = state.get("memories") or []
        if memories:
            memory_summary = "\n".join([m["content"] for m in memories])
        else:
            memory_summary = "None"
        print("memory_summary in planner", memory_summary)

        num_retrieved = len(retrieved_chunks)
        num_reranked = len(reranked_chunks)
        observation_text = "\n".join(observations) if observations else "None"

        # --- Memory Summaries ---
        
        # --- System Prompt ---
        REASONING_PLANNER_PROMPT = """You are the Reasoning Engine of an Agentic RAG system.

Your task is to decide the SINGLE next logical action based on the current state.
You do NOT answer the user directly unless instructed.

AVAILABLE TOOLS:
1. retrieve
   Use this to fetch documents from the knowledge base.
   Args: {"user_query_expansion": bool}

2. rerank
   Use this after retrieval to prioritize relevant chunks.

3. answer
   Use this ONLY if the current information is sufficient to answer the query.

4. stop
   Use this if multiple attempts failed or the query cannot be answered.

DECISION RULES:
- Memory Rule: 
    - If memory clearly contains the answer, and you find it to be eough to aswer the query then go to answer and add args -> "source": "memory" other else "source": "rag"
    - if there is no need to retrieve or rerank to then dont retrieve or rerank, go to answer. Otherwise, go to retrieve.
 (if {memory_summary} is empety then skip this "Memory Rule:" rule)
    
- If no chunks are retrieved → retrieve (user_query_expansion: true)
- If chunks exist but are unordered or noisy → rerank
- If reranked chunks clearly contain the answer → answer
- If repeated attempts fail → stop

OUTPUT FORMAT (STRICT JSON ONLY):
{
  "thought": "brief reasoning",
  "tool": "retrieve | rerank | answer | stop",
  "args": { }
}

Do NOT return anything outside this JSON.
Do NOT use markdown.
"""

        # --- User Prompt (State Injection) ---
        user_prompt = f"""
USER QUERY:
{user_query}
PAST MEMORIES:
{memory_summary}

CURRENT STATE:
- Retrieved Chunks: {num_retrieved}
- Reranked Chunks: {num_reranked}
- Observations:
{observation_text}
"""

        config = GenerateContentConfig(
            system_instruction=REASONING_PLANNER_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
        )

        try:
            response = self.llm_client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )
            print("\n\nReasoning Planner Response:\n")
            print(response.text)
            print("=" * 50)
            return json.loads(response.text)

        except Exception:
            # Hard safety fallback to avoid agent crash
            return {
                "thought": "Planner failed or returned invalid output",
                "tool": "stop",
                "args": {}
            }
