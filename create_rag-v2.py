from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_core.documents import Document

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())
print("Start RAG")

# Step 1: Load raw PDF(s)
DATA_PATH="data/"
DB_FAISS_PATH = "vectorstore/rekso_2023_2025_v2"


# Step 0: Custom Pasal-based Splitter
class PasalSplitter(TextSplitter):
    """Custom splitter by 'Pasal X' pattern."""
    def __init__(self, separator=r"(Pasal\s+\d+)", **kwargs):
        super().__init__(**kwargs)
        import re
        self.separator = separator
        self.pattern = re.compile(separator)

    def split_text(self, text):
        import re
        splits = re.split(self.pattern, text)
        result = []
        current_chunk = ""

        for part in splits:
            if re.match(self.pattern, part):
                if current_chunk:
                    result.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += f" {part}"

        if current_chunk:
            result.append(current_chunk.strip())

        return result


# Step 1: Load PDF files and clean text
def clean_text(text):
    """Preprocessing: Remove extra spaces, line breaks, and normalize text."""
    cleaned_text = text.replace('\n', ' ').strip()
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        # Tambahkan metadata source kalau belum ada
        doc.metadata["source"] = doc.metadata.get("source", "Unknown PDF")

    print(f"✅ Loaded {len(documents)} documents")
    return documents

documents = load_pdf_files(DATA_PATH)

# Step 2 chunking doc
def create_chunks(docs):
    splitter = PasalSplitter()
    chunks = []

    for doc in docs:
        split_texts = splitter.split_text(doc.page_content)
        for text in split_texts:
            if text.strip():
                # preserve metadata per chunk
                new_doc = doc.__class__(page_content=text, metadata=doc.metadata)
                chunks.append(new_doc)

    print(f"✅ Created {len(chunks)} Pasal-based chunks")
    return chunks

text_chunks = create_chunks(documents)

# Step 3: Create Vector Embeddings 

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Create or update FAISS vector database
def build_or_update_faiss(db_path, text_chunks, embedding_model):
    if os.path.exists(db_path):
        print("📦 Existing FAISS DB found — adding new chunks")
        db = FAISS.load_local(db_path, embedding_model)
        db.add_documents(text_chunks)
    else:
        print("📦 No FAISS DB found — creating new one")
        db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(db_path)
    print(f"✅ FAISS DB saved at {db_path}")

build_or_update_faiss(DB_FAISS_PATH, text_chunks, embedding_model)

print("🎉 Improved RAG Pipeline Completed")