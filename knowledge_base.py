import os
import numpy as np
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFReader
import openai
import pickle
from dotenv import load_dotenv
from lance import LanceDb
from lance import SearchType

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Path to the local folder containing PDF files
pdf_folder_path = "CEO_Pdf"  # Replace this with the actual folder path

# Extract PDF texts from a folder
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]

# Function to extract text from PDF using PyMuPDF (fitz)
def extract_text_from_pdfs(pdf_files):
    text_data = []
    for pdf_file in pdf_files:
        try:
            pdf_reader = PDFReader(pdf_file, chunk=True)
            text = pdf_reader.extract_text()
            text_data.append(text)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    return text_data

# Extract text from all PDFs in the folder
pdf_texts = extract_text_from_pdfs(pdf_files)

# Generate embeddings using OpenAI
embedder = OpenAIEmbedder(model="text-embedding-ada-002")
embeddings = []

for text in pdf_texts:
    embedding = embedder.encode(text)
    embeddings.append(embedding)

# Convert the embeddings list to a NumPy array for LanceDb
embeddings = np.array(embeddings).astype("float32")

# Initialize LanceDb
lancedb = LanceDb(
    table_name="pdf_documents",  # Name of the table to store the embeddings
    uri="tmp/lancedb",           # Path to store the LanceDb index
    search_type=SearchType.vector,  # We will use vector search
    embedder=OpenAIEmbedder(model="text-embedding-ada-002"),  # Same embedder for both encoding and indexing
)

# Add embeddings to LanceDb
lancedb.add(embeddings)

# Optionally, save the LanceDb index to disk (this happens automatically)
lancedb.save()

# Save the PDF texts corresponding to the embeddings for reference
with open("pdf_texts.pkl", "wb") as f:
    pickle.dump(pdf_texts, f)

# Function to search the LanceDb index
def search_lancedb(query, top_k=5):
    query_embedding = embedder.encode(query)
    query_embedding = np.array([query_embedding]).astype("float32")
    _, indices = lancedb.search(query_embedding, top_k)  # Search the top_k nearest neighbors
    return indices

# Create the agent with the knowledge base (LanceDb vector search)
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    knowledge=lancedb,  # Use LanceDb as the knowledge source
    show_tool_calls=True,
    markdown=True,
)

# Query the agent with a question
response = agent.print_response("How do I make chicken and galangal in coconut milk soup?", stream=True)

# Print the response
print(response)
