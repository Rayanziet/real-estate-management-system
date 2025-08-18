import os
import shutil
from xml.dom.minidom import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import get_chroma_db

CHROMA_PATH = "chroma"
DATA_PATH = "data/csv_data"


def load_csv_documents(csv_dir):
    all_docs = []
    for filename in os.listdir(csv_dir):
        full_path = os.path.join(csv_dir, filename)
        loader = CSVLoader(full_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

#no need to split the csv files, as they are already structured and the columns are up to 4-5 columns only


#this is used to create an identifier so the db won't store duplicates
def calculate_row_ids(docs):
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        row_num = doc.metadata.get("row", i)
        source_name = os.path.basename(source) if source else "csv"
        doc.metadata["id"] = f"{source_name}:row-{row_num}"
    return docs

def add_to_chroma(docs: list[Document]):
    db = get_chroma_db()
    docs_with_ids = calculate_row_ids(docs)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_docs = [doc for doc in docs_with_ids if doc.metadata["id"] not in existing_ids]

    if len(new_docs):
        new_ids = [doc.metadata["id"] for doc in new_docs]
        db.add_documents(new_docs, ids=new_ids)
        db.persist()
    else:
        print("No new rows to add.")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def csv_pipeline():
    csv_docs = load_csv_documents(DATA_PATH)
    add_to_chroma(csv_docs)