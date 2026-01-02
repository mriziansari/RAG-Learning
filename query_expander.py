from google.genai.types import GenerateContentConfig

class QueryExpander:
    def __init__(self, llm_client, model_name: str):
        self.llm_client = llm_client
        self.model = model_name

    def expand_query(self, query_text: str, n: int = 3) -> list[str]:
        """Expands the query using the LLM."""

        user_prompt = f"""
        Rewrite the following question into {n} different,
        precise, information-seeking search queries.
        Do not add new facts.
        

        Original question:
        {query_text}
        """
        system_prompt = f"""
        You are a query expander. Your goal is to expand the query into multiple different,
        precise, information-seeking search queries. Do not add new facts. 
        Just give direct queries. Dont add any thing before or after the query.
        Return exactly {n} queries.
        Each query on a new line.
        No numbering.
        No bullets.
        """
        try:
            config = GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3
            )
            response = self.llm_client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config
            )

            return self._parse(response.text)

        except Exception as e:
            raise ValueError(f"Error expanding query: {e}")

    def _parse(self, text: str) -> list[str]:
        """
        Parse the response text into a list of queries.
        """
        try:
            if not text:
                raise ValueError("No text provided")
            
            return [line.strip("-").strip() for line in text.splitlines() if line.strip()]

        except Exception as e:
            raise ValueError(f"Error parsing query: {e}")


    def is_query_weak(self, query: str) -> bool:
        tokens = query.strip().split()
        return len(tokens) < 10