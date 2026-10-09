
from rag_service import get_answer

question = input("Apna Teckinfo support question likho: ")

print("\nRAG answer generate kar raha hai...")
answer = get_answer(question)

print("\n========== RAG ANSWER ==========")
print(answer)
print("================================")