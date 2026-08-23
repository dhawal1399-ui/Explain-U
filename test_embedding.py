from sentence_transformers import SentenceTransformer
import chromadb
import ollama

model = SentenceTransformer("all-MiniLM-L6-v2")

vectors = model.encode("Hello")

print(vectors,vectors.shape)
client = chromadb.Client()
collection = client.get_or_create_collection(name="documents")

collection.add(
    ids = ["1"],
    documents=["hello"],
    embeddings=[vectors.tolist()]
)
result = collection.get()
print(result)

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": """
Use only the provided context.

Context:
Dhawal Somkuwar
Phone: +91-914-653-7288
Professional Summary:
Python Backend Developer with 3+ years of experience.

Question:
What is the candidate full name?

Answer only the name.
"""
        }
    ]
)

print(response["message"]["content"])