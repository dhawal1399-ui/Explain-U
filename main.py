from fastapi import FastAPI, UploadFile, File, Form
from pypdf import PdfReader
import io
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
from pydantic import BaseModel

app = FastAPI()

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
    print("\n========== QUERY REWRITING ==========")
    print("Original Query:", question)
    print("Rewritten Query:", final_quest)
    print("=====================================")
    result = collection.query(
        query_texts=[final_quest],
        n_results=3
    )

    docs = result["documents"][0]

    print("DOC COUNT:", len(docs))

    for i, doc in enumerate(docs):
        print(f"\n--- DOC {i} ---")
        print(doc)

    context = "\n---CHUNK---\n".join(docs)

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

