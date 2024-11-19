import streamlit as st
from PyPDF2 import PdfReader

st.title('AI-Study')

upload_type = st.radio("Upload type:", ("Single File", "Multiple Files"), horizontal=True)

def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

if upload_type == "Single File":
    uploaded_file = st.file_uploader("Choose a file to upload")
    if uploaded_file:
        st.write("Filename:", uploaded_file.name)
        st.write("File type:", uploaded_file.type)
        st.write("File size:", uploaded_file.size)
        text = read_pdf(uploaded_file)
        print(text)
else:
    uploaded_files = st.file_uploader("Choose files to upload", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            st.write("Filename:", file.name)
            st.write("File type:", file.type)
            st.write("File size:", file.size)
            text = read_pdf(file)
            print(text)

prompt = st.chat_input("Input some text here")
if prompt:
    st.write(f"The following text has been written in the prompt: {prompt}")
