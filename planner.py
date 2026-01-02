import json
from google.genai.types import GenerateContentConfig


class Planner:
    def __init__(self, llm_client, model_name: str):
        self.llm_client = llm_client
        self.model = model_name
        
    def plan(self, query_text: str):
        """Plan the query using the LLM.
        Plan about which tools need to be used to answer the query.
        Available tools are:
        1. Retriever (args: user_query_expansion: True/False)
        2. Reranker
        3. Response Generator
        Return the plan in json format.
        """
        system_prompt = """ou are the Strategic Planner for an Agentic RAG system.
Your role is to analyze a user query and generate a structured execution plan.
        Given a user query, decide:
        1) The intent
        2) Which tools to call
        3) Whether query expansion is needed

        Available tools:
- retrieve (args: use_query_expansion: true/false)
- rerank
- answer

Rules:
- Always include retrieve → rerank → answer in this order
- Use query expansion only if the query is vague or short
- Return ONLY valid JSON, As mentioned in the output format. Do not include any other text.
- No explanations

OUTPUT FORMAT:
{
  "intent": "fact_lookup",
  "actions": [
    {
      "tool": "retrieve",
      "args": { "user_query_expansion": true }
    },
    { "tool": "rerank", "args": {} },
    { "tool": "answer", "args": {} }
  ]
}
"""
        
        user_prompt = f"""{query_text}"""
        config = GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.0,
        )
        try:
            response = self.llm_client.models.generate_content(
                model=self.model,
                config=config,
                contents=user_prompt,
            )
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return f"Invalid JSON response from Planner. Raw response: {response.text}"
        except Exception as e:
            return f"Error generating content: {e}" 