import fitz
import textwrap
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from transformers import LlamaTokenizer, LlamaForCausalLM

pdf_path = './insurance.pdf'
doc = fitz.open(pdf_path)
text = "".join(page.get_text() for page in doc)
text = text.replace('\u200b', ' ')

# Text Chunking
chunk_size = 750
chunks = textwrap.wrap(text, chunk_size)

# Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks).tolist()


# Qdrant Configuration
api_key = "ETlTa7qqoQWLToVpMTgLECxSsrEXuT7fpAbrg97QEHvYvyBhYPOpcw"
qdrant_url = "https://2bf18249-d687-4b2e-9f7e-2eeba4f613c7.europe-west3-0.gcp.cloud.qdrant.io:6333/"
collection_name = "insurance_collection_doc_3"
vector_size = 384
vector_params = VectorParams(size=vector_size, distance=Distance.COSINE)
client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=60)

try:
    # Check if the collection exists
    existing_collections = client.get_collections()
    collection_names = [collection.name for collection in existing_collections.collections]
    if collection_name not in collection_names:
        client.create_collection(collection_name=collection_name, vectors_config=vector_params)
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")
except Exception as e:
    print(f"Error occurred while checking/creating collection: {e}")


# Insert Data
points = [
    PointStruct(id=idx, vector=embedding, payload={"content": chunk})
    for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
]
client.upsert(collection_name=collection_name, points=points)

# Query Embedding
query_text = "Who is Virat Kohli?"
query_embedding = model.encode([query_text]).tolist()[0]
results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=3
)

# Extract Context
context = "".join(result.payload["content"] for result in results)
print(context)
# Hugging Face Login
login(token="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")

# Load LLaMA Model
model_name = "meta-llama/Llama-2-7b-hf"  # Ensure the model exists
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Generate Response
prompt = f"Context: {context}\n\nQuery: {query_text}\n\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Response:", response)
