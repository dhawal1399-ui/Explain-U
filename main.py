from fastapi import FastAPI, UploadFile, File, Form
from pypdf import PdfReader
import io
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import ollama
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

app = FastAPI()

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class ChatRequest(BaseModel):
    question: str
    session_id: str

conversation_history = {}


@app.get("/")
def read_root():
    return 'Hello world'

@app.post("/upload")
async def upload_file(

    file : UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form(...)

    ):

    if session_id not in conversation_history:
        conversation_history[session_id] = []

    history = conversation_history[session_id]

    file_content = await file.read()
    reader = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = chunk_text(text, 500, 100)

    tokenized_chunks = [chunk.split() for chunk in chunks]

    bm25 = BM25Okapi(tokenized_chunks) 
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks)
    client = chromadb.Client()

    collection = client.get_or_create_collection(name="documents")


    collection.add(
        ids = [str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors
    )

    final_quest = rewrite_query(question, history)
    # print("\n========== QUERY REWRITING ==========")
    # print("Original Query:", question)
    # print("Rewritten Query:", final_quest)
    # print("=====================================")
    result = collection.query(
        query_texts=[final_quest],
        n_results=10
    )

    docs = result["documents"][0]

    tokenized_query = final_quest.split()

    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_indexes = bm25_scores.argsort()[-5:][::-1]

    bm25_docs = [chunks[i] for i in top_bm25_indexes]

    print("\n========== BM25 TOP RESULTS ==========")

    for i, doc in enumerate(bm25_docs):
        print(f"\n--- BM25 DOC {i} ---")
        print(doc)

    print("======================================")

    combined_docs = list(dict.fromkeys(docs + bm25_docs))

    print("\n========== CHROMA TOP RESULTS ==========")

    for i, doc in enumerate(combined_docs):
        print(f"\n--- CHROMA DOC {i} ---")
        print(doc)

    print("========================================")
    pairs = [[final_quest, doc] for doc in combined_docs]

    scores = reranker.predict(pairs)

    for i, score in enumerate(scores):
        print(f"Chunk {i} Score: {score}")

    ranked_docs = sorted(
        zip(scores, combined_docs),
        key=lambda x: x[0],
        reverse=True
    )

    

    top_docs = [doc for score, doc in ranked_docs[:8]]

    context = "\n---CHUNK---\n".join(top_docs)

    print("\n========== RERANKED TOP 5 ==========")

    for i, doc in enumerate(top_docs):
        print(f"\n--- TOP DOC {i} ---")
        print(doc)

    print("====================================")

    response = ollama.chat(
            model="llama3.2",
            messages=history + [
                {
                    "role": "user",
                    "content": f"""
        You are a document question-answering assistant.

        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {question}

        Answer the user's question based only on the provided context and conversation history.

        If the answer is not available in the provided context, say that you could not find the answer.
        Do not make up information.
        """
                }
            ]
        )

    history.append({
        "role": "user",
        "content":question
    })
    
    history.append({
        "role": "assistant",
        "content": response["message"]["content"]
    })

    print("\n========== HISTORY AFTER RESPONSE ==========")
    print(history)
    print("============================================")

    return {
        "filename":file.filename,
        "text":text,
        "chunks":chunks,
        # "result":result,
        "response":response["message"]["content"]
    }


def rewrite_query(question, history):

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    prompt = f"""
You are a query rewriting assistant.

Given the conversation history and the user's current question,
rewrite the current question into a standalone question.

Rules:
- Use the conversation history to resolve references like "he", "his", "it", "they", etc.
- Do not answer the question.
- Return ONLY the rewritten question.

Conversation History:
{history_text}

Current Question:
{question}
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

def chunk_text(text, chunk_size = 100, overlap = 20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if not chunk : 
            break
        start = end - overlap

    return chunks

