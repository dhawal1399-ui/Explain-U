from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import io
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


app = FastAPI()

@app.get("/")
def read_root():
    return 'Hello world'

@app.post("/upload")
async def upload_file(file : UploadFile = File(...)):
    file_content = await file.read()
    reader = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = chunk_text(text, 100, 20)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks)
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="documents")

    collection.add(
        ids = [str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors
    )

    result = collection.query(
        query_texts=["What is the candidate full name?"],
        n_results=3
    )

    docs = result["documents"][0]

    print("DOC COUNT:", len(docs))

    for i, doc in enumerate(docs):
        print(f"\n--- DOC {i} ---")
        print(doc)

    context = "\n---CHUNK---\n".join(docs)

    print("context:",context)
    response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": f"""
        You are a document question-answering assistant.

        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        What is the candidate's full name?

        The candidate's full name is explicitly shown at the beginning of the context.
        Return only the full name.
        """
                }
            ]
        )

    return {
        "filename":file.filename,
        "text":text,
        "chunks":chunks,
        # "result":result,
        "response":response["message"]["content"]
    }



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






