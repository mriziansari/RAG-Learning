

class ToolRegistry:
    """Registry for tools used in the executor. This class is the part of agentic RAG system.
    Args:
        query_expander: QueryExpander
        retriever: Retriever
        reranker: Reranker
        response_generator: ResponseGenerator
    """
    def __init__(self, query_expander, retriever, reranker, response_generator, memory_store):
        self.query_expander = query_expander
        self.retriever = retriever
        self.reranker = reranker
        self.response_generator = response_generator
        self.memory_store = memory_store

    def retrieve(self, state, user_query_expansion: bool):
        if user_query_expansion:
            expended_queries = self.query_expander.expand_query(state["user_query"])
        else:
            expended_queries = [state["user_query"]]
        
        state["expanded_queries"] = expended_queries
        state["retrieved_chunks"] = self.retriever.retrive_chunks(expended_queries)
        
    def rerank(self, state):
        stage_1_chunks = self.reranker.stage1_reranking(state["user_query"], state["retrieved_chunks"], 5)
        stage_2_chunks = self.reranker.stage2_reranking(state["user_query"], stage_1_chunks)
        state["reranked_chunks"] = stage_2_chunks

    def answer(self, state):

        # If source is memory then get the answer base on that memory else get the answer base on reranked chunks
        source = state.get("answer_source", "rag")
        if source == "memory":
            context = "\n".join(m["content"] for m in state["memories"])
        else:
            context = state.get("reranked_chunks", [])
        
        answer_dict = self.response_generator.generate_response(state["user_query"], context, source = source)

        if not answer_dict:
            state["observations"].append("Empty answer generated")
            return None
        
        state["answer"] = answer_dict

        # Save memory ONLY if answer came from source rag or improved memory
        if source == "rag":
            memory_summary = answer_dict.get("memory_summary", "")

            if memory_summary:
                self.memory_store.add(
                    memory_summary,
                    memory_type="semantic",
                metadata={
                "query": state["user_query"]
            })
            else:
                state["observations"].append("Memory summary is empty")
        return answer_dict
    
