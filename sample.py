import os
import fitz
import qdrant_client
from dotenv import load_dotenv
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from test import get_groq_response 
from paraphrase import get_paraphrase
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

from transformers import T5Tokenizer, T5ForConditionalGeneration

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


qdrant_client = QdrantClient(
    url="https://2bf18249-d687-4b2e-9f7e-2eeba4f613c7.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="n5VmO7yZHYhBRjdcCY-ERpuouGk8kR10DuLKSIILiL9k6AtPYFHAyw", timeout = 60
)



collection_name = "google_doc_embeddings"
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

pdf_path = "insurance.pdf"
doc = fitz.open(pdf_path)

def split_into_chunks_by_words(text, words_per_chunk=500):
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) == words_per_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

doc_content = ""
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    doc_content += page.get_text()

chunks = split_into_chunks_by_words(doc_content, words_per_chunk=500)
chunk_embeddings = embeddings.embed_documents(chunks)

points = [
    {"id": idx + 1, "vector": embedding, "payload": {"text": chunk}}
    for idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings))
]
qdrant_client.upsert(collection_name=collection_name, points=points)

def construct_query(query):
    paraphrased_text = get_paraphrase(query)
    return f"{paraphrased_text}"

def generate_hypothetical_doc(question):
    prompt = f"Write a hypothetical context based on the question: {question}"
    response = get_groq_response(question)
    return response

def keyword_search(query, chunks):
    return [chunk for chunk in chunks if query.lower() in chunk.lower()]

def hybrid_retrieval_with_hyde(query, chunks, k=5):
    hypothetical_doc = generate_hypothetical_doc(query)
    hypothetical_embedding = embeddings.embed_documents([hypothetical_doc])[0]
    query_embedding = embeddings.embed_documents([query])[0]

    dense_results_query = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k
    )
    dense_results_hypothetical = qdrant_client.search(
        collection_name=collection_name,
        query_vector=hypothetical_embedding,
        limit=k
    )

    sparse_results = keyword_search(query, chunks)
    combined_results = dense_results_query + dense_results_hypothetical + sparse_results
    return combined_results[:k]

def rerank_results(query_embedding, results):
    result_texts = [result.payload["text"] for result in results]
    result_embeddings = embeddings.embed_documents(result_texts)
    similarities = cosine_similarity([query_embedding], result_embeddings)
    ranked_results = [result for _, result in
                      sorted(zip(similarities[0], results), reverse=True)]
    return ranked_results

def process_question(question): 
    query = question[len(question) - 1]["text"]
    constructed_query = construct_query(query)
    query_embedding = embeddings.embed_documents([constructed_query])[0]
    dense_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5
    )

    reranked_docs = rerank_results(query_embedding, dense_results)
    context = " ".join([doc.payload["text"] for doc in reranked_docs])
    response = get_groq_response(f"Context: {context}\n\nQuestion: {question}\n\nAnswer: . Answer from the given context only.")
    structured_output = {
        "Question": question,
        "Constructed Query": constructed_query,
        "Answer": response,
        "Relevant Documents": [doc.payload["text"] for doc in reranked_docs],
        "Hypothetical Document": "",
    }
    return structured_output


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

@app.post("/items/")
def create_item(item: dict):
    question = item["ChatHistory"]
    structured_output = process_question(question)
    return structured_output

# print(process_question("What is a car insurance?"))