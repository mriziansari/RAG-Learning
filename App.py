from google.genai import Client 
from dotenv import load_dotenv
import os
from query_expander import QueryExpander
from retriever import Retriever, Reranker, QueryEmbedder
from generation import ResponseGenerator


load_dotenv()
if __name__ == "__main__":
    gclient = Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model_name = "gemini-2.5-flash" 

    query_expander = QueryExpander(gclient, model_name)
    retriever = Retriever()

    query_text = "By how many tons per capita must the urban carbon footprint drop to meet the 2035 target from the 2024 baseline?"
    # query_text = "How does the 'Right to Shade' regulation in Phoenix potentially impact the temperature of neighboring buildings?"
    # query_text = "Identify the transit mode that is the most energy-efficient according to Table 1 and explain its fire rating if mentioned?"
    # query_text = "The report mentions that Autonomous Vehicles eliminate traffic jams. Is this consistent with the findings in San Francisco?"
    # query_text = "What is the specific initial cost barrier mentioned for Kinetic Pavements, and which city conducted the pilot study?"

    if query_expander.is_query_weak(query_text):
        expanded_queries = query_expander.expand_query(query_text)
    else:
        expanded_queries = [query_text]

    print("\nExpanded Queries:")
    for i, query in enumerate(expanded_queries):
        print(f"\n Query {i+1}: {query}")

    
    retrieved_chunks = retriever.retrive_chunks(expanded_queries)
   
    # rerank chunks
    reranker = Reranker()
    stage1_retrieved_chunks = reranker.stage1_reranking(query_text, retrieved_chunks, 4)
    stage2_retrieved_chunks = reranker.stage2_reranking(query_text, stage1_retrieved_chunks)
    # generate response
    response_generator = ResponseGenerator(gclient)
    response = response_generator.generate_response(query_text, stage2_retrieved_chunks)
    print(f"\nQuery: {query_text}")
    print(f"\nResponse: {response}")
   
   


    
