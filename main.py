import os
import re
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader

st.title('AI-Study')

upload_type = st.radio("Upload type:", ("Single File", "Multiple Files"), horizontal=True)

# ensures a data folder exists
if not os.path.exists("data"):
    os.makedirs("data")

def store_txt(content):
    '''
    For storing extracted text from PDFs locally
    '''
    dir_name = os.path.join("data", "upload_history")
    file_name = os.path.join(dir_name, "extracted_text.txt")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(content)

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\n+', '\n', text) # remove unnecessary newlines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text) # ensures proper paragraph splitting
    text = re.sub(r'\s+', ' ', text) # removes unnecessary spaces
    text = re.sub(r'[^\x20-\x7E]+', '', text) # removes non-ASCII characters
    return text

def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        text = clean_text(text)
    return text

if upload_type == "Single File":
    uploaded_file = st.file_uploader("Choose a file to upload")
    if uploaded_file:
        st.write("Filename:", uploaded_file.name)
        st.write("File type:", uploaded_file.type)
        st.write("File size:", uploaded_file.size)
        text = read_pdf(uploaded_file)
        print(text)
        store_txt(text + "\n")
else:
    uploaded_files = st.file_uploader("Choose files to upload", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            st.write("Filename:", file.name)
            st.write("File type:", file.type)
            st.write("File size:", file.size)
            text = read_pdf(file)
            text = text.strip()
            print(text)
            store_txt(text + "\n")

prompt = st.chat_input("Input some text here")
if prompt:
    st.write(f"The following text has been written in the prompt: {prompt}")
