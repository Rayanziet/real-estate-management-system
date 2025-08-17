import os
import shutil
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data/pdf_data"


def load_pdf_documents(pdf_dir):
    all_docs = []
    for filename in os.listdir(pdf_dir):
        full_path = os.path.join(pdf_dir, filename)
        loader = UnstructuredPDFLoader(full_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return text_splitter.split_documents(documents)


def embed_docs():
    embedder = HuggingFaceEmbeddings(model="BAAI/bge-m3")
    return embedder

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embed_docs()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[]) 
    existing_ids = set(existing_items["ids"])
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")

#this is used to create an identifier so the db won't store duplicates
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

# Ref: https://www.youtube.com/watch?v=2TJxpyO3ei4&t=946s

def pdf_pipeline():
    pdf_docs = load_pdf_documents(DATA_PATH)
    split_docs = split_documents(pdf_docs)
    add_to_chroma(split_docs)