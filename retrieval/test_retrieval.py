from retrieval.search.retriever import Retriever


retriever = Retriever()

results = retriever.retrieve(
    "What are symptoms of tuberculosis?",
    top_k=5
)


for i, result in enumerate(results):

    print("\n====================")
    print("Rank:", i+1)
    print("Score:", result["score"])
    print("Metadata:")
    print(result["metadata"])

    print("\nText:")
    print(result["text"][:300])