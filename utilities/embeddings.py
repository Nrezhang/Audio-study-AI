import re 
from transformers import AutoModel


# Initialize the model
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into smaller chunks of a specified size."""
    words = text.split()
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i:i + chunk_size])

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\n+', '\n', text) # remove unnecessary newlines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text) # ensures proper paragraph splitting
    text = re.sub(r'\s+', ' ', text) # removes unnecessary spaces
    text = re.sub(r'[^\x20-\x7E]+', '', text) # removes non-ASCII characters
    return text

def generate_embeddings(text, chunk_size=500, overlap=50):
    text = clean_text(text)
    chunks = list(chunk_text(text, chunk_size=chunk_size, overlap=overlap))
    embeddings = [model.encode(chunk, task="text-matching") for chunk in chunks]
    return embeddings, chunks

# def test():
#     test_embedding = model.encode("What is the school gym described as?", task="text-matching")
#     return test_embedding