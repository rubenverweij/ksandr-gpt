from langchain_community.graphs import Neo4jGraph
from langchain.retrievers import GraphCypherQAChain
from main import LLM
import argparse

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
chain = GraphCypherQAChain.from_llm(
    LLM,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,  # alleen bij testen
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="question")
    args = parser.parse_args()
    response = chain.invoke({"query": args.question})
    print(response["result"])
