import os
import re
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader
from utilities.embeddings import generate_embeddings, test
import json
import chromadb

chroma_client = chromadb.PersistentClientClient(path="./data/chroma")
collection = chroma_client.get_or_create_collection(name="pdf_embeddings")


st.title('AI-Study')

upload_type = st.radio("Upload type:", ("Single File", "Multiple Files"), horizontal=True)

# ensures a data folder exists
if not os.path.exists("data"):
    os.makedirs("data")

def store_txt(content, filename):
    '''
    For storing extracted text from PDFs locally
    '''
    dir_name = os.path.join("data", "upload_history")
    file_name = os.path.join(dir_name, filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(content)

# def clean_text(text):
#     text = text.strip()
#     text = re.sub(r'\n+', '\n', text) # remove unnecessary newlines
#     text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text) # ensures proper paragraph splitting
#     text = re.sub(r'\s+', ' ', text) # removes unnecessary spaces
#     text = re.sub(r'[^\x20-\x7E]+', '', text) # removes non-ASCII characters
#     return text

def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def query_db(query):
    query_embedding, _ = generate_embeddings(query)
    results = collection.query(query_embeddings=[query_embedding[0]], num_results=3)
    return results

if upload_type == "Single File":
    uploaded_file = st.file_uploader("Choose a file to upload")
    if uploaded_file:
        st.write("Filename:", uploaded_file.name)
        st.write("File type:", uploaded_file.type)
        st.write("File size:", uploaded_file.size)
        text = read_pdf(uploaded_file)
        store_txt(text + "\n", f"{uploaded_file.name}.txt")

        embeddings, chunks = generate_embeddings(text)
        chunked_file = f"chunked_{uploaded_file.name}.txt"
        store_txt("\n\n".join(chunks), chunked_file)

        embeddings_json = json.dumps([emb.tolist() for emb in embeddings])
        store_txt(embeddings_json, f"embeddings_{uploaded_file.name}.json")
        
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            collection.add(
                ids=[f"{uploaded_file.name}_chunk_{i}"],  # Unique ID for each chunk
                embeddings=[embedding.tolist()],          # Embedding vector
                metadatas=[{"chunk": chunk, "document": uploaded_file.name}]  # Metadata
            )
        
        st.write(f"Embeddings for {uploaded_file.name} stored in ChromaDB.")
else:
    uploaded_files = st.file_uploader("Choose files to upload", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            st.write("Filename:", file.name)
            st.write("File type:", file.type)
            st.write("File size:", file.size)
            text = read_pdf(file)
            text = text.strip()
            store_txt(text + "\n", f"{file.name}.txt")

            embeddings, chunks = generate_embeddings(text)
            chunked_file = f"chunked_{file.name}.txt"
            store_txt("\n\n".join(chunks), chunked_file)

            embeddings_json = json.dumps([emb.tolist() for emb in embeddings])
            store_txt(embeddings_json, f"embeddings_{file.name}.json")

            for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
                collection.add(
                    ids=[f"{file.name}_chunk_{i}"],  # Unique ID for each chunk
                    embeddings=[embedding.tolist()], # Embedding vector
                    metadatas=[{"chunk": chunk, "document": file.name}]  # Metadata
                )
            
            st.write(f"Embeddings for {file.name} stored in ChromaDB.")            

# test_embedding = test()
# print(test_embedding)

prompt = st.chat_input("Input some text here")
if prompt:
    st.write(f"The following text has been written in the prompt: {prompt}")
    results = query_db(prompt)
    st.write("Top matches: ")
    for result in results["metadatas"]:
        st.write(f"Document: {result['document']}, Chunk: {result['chunk']}")

