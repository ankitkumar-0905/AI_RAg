
import chromadb
import torch
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Project paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = Path(
    os.getenv(
        "CHROMA_PATH",
        r"D:\Ankit\teckinfo_RAG\chroma_db"
    )
)
# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name="teckinfo_support")

# Load existing LLM
print("Loading Qwen model...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name)
llm.eval()


def get_answer(query: str) -> str:
    """Retrieve relevant Teckinfo context and generate an answer."""

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    documents = results.get("documents", [[]])[0]

    if not documents or not documents[0]:
        return "Information not available in the provided documentation."

    context = documents[0]

    prompt = f"""You are a technical support assistant for Teckinfo.
Answer using ONLY the information in the context.
Do not guess or use outside knowledge.
If the answer is missing, say:
Information not available in the provided documentation.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    # Decode only newly generated tokens, not the original prompt
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return answer or "Sorry, I could not generate an answer."