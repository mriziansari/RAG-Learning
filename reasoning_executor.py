class ReasoningExecutor:
    def __init__(self,planner, tool_registry, max_steps: int = 3):
        self.tools = tool_registry
        self.planner = planner
        self.max_steps = max_steps
        # Tool limits
        self.max_retrieve = 2
        self.max_rerank = 1

    def run(self, user_query: str):
        """Get the plan and execute the tools in sequence according to the plan
        args:
            user_query: str (user query)
        """
        state = {
            "user_query": user_query,
            "expanded_queries": None,
            "retrieved_chunks": None,
            "reranked_chunks": None,
            "answer": None,
            "observations": [],
            "tool_calls": {
                "retrieve": 0,
                "rerank": 0,
                "answer": 0
            }
        }

        for step in range(self.max_steps):
            action = self.planner.next_action(state)
            
            tool = action["tool"]
            args = action.get("args", {})
            # loop Guard
            if tool == "retrieve" and state["tool_calls"]["retrieve"] >= self.max_retrieve:
                state["observations"].append("Max retrieve calls reached")
                tool = "stop"

            if tool == "rerank" and state["tool_calls"]["rerank"] >= self.max_rerank:
                state["observations"].append("Max rerank calls reached")
                tool = "answer"

                #Execution
            if tool == "retrieve":
                state["tool_calls"]["retrieve"] += 1
                self.tools.retrieve(state, **args)
                print("Retriver is executed")
                state["observations"].append(
                    f"Retrieved {len(state['retrieved_chunks'])} chunks"
                )

            elif tool == "rerank":
                state["tool_calls"]["rerank"] += 1
                self.tools.rerank(state)
                print("Reranker is executed")
                state["observations"].append(
                    f"Reranked {len(state['reranked_chunks'])} chunks"
                )
            elif tool == "answer":
                state["tool_calls"]["answer"] += 1
                self.tools.answer(state)
                print("Answerer is executed")
                return state["answer"]

            elif tool == "stop":
                return "Unable to answer with available information"

            else:
                raise ValueError(f"Unknown tool: {tool}")

        return "Stopped due to step limit"
