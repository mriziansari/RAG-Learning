


class ToolRegistry:
    def __init__(self, query_expander, retriever, reranker, response_generator):
        self.query_expander = query_expander
        self.retriever = retriever
        self.reranker = reranker
        self.response_generator = response_generator

    def retrieve(self, state, user_query_expansion: bool):
        if user_query_expansion:
            expended_queries = self.query_expander.expand_query(state["query"])
        else:
            expended_queries = [state["query"]]
        
        state["expanded_queries"] = expended_queries
        state["retrieved_chunks"] = self.retriever.retrive_chunks(expended_queries)
        
    def rerank(self, state):
        stage_1_chunks = self.reranker.stage1_reranking(state["query"], state["retrieved_chunks"], 5)
        stage_2_chunks = self.reranker.stage2_reranking(state["query"], stage_1_chunks)
        state["reranked_chunks"] = stage_2_chunks

    def answer(self, state):
        state["answer"] = self.response_generator.generate_response(state["query"], state["reranked_chunks"])
    
