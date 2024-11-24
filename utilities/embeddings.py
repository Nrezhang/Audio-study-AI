import re 
from transformers import AutoModel


# Initialize the model
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

def chunk_text(text, chunk_size=300):
    """Split text into smaller chunks of a specified size."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\n+', '\n', text) # remove unnecessary newlines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text) # ensures proper paragraph splitting
    text = re.sub(r'\s+', ' ', text) # removes unnecessary spaces
    text = re.sub(r'[^\x20-\x7E]+', '', text) # removes non-ASCII characters
    return text

def generate_embeddings(text):
    text = clean_text(text)
    chunks = list(chunk_text(text))
    embeddings = [model.encode(chunk, task="text-matching") for chunk in chunks]
    return embeddings, chunks