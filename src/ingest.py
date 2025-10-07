from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, sys, glob

ROOT = Path(__file__).resolve().parents[1]
DATA_BASE = ROOT / "data"
PDF_GLOB = DATA_BASE / "pdf" / "**" / "*.pdf"
TXT_GLOB = DATA_BASE / "txt" / "**" / "*"

def load_docs():
    docs = []
    pdf_paths = glob.glob(str(PDF_GLOB), recursive=True)
    txt_paths = [p for p in glob.glob(str(TXT_GLOB), recursive=True)
                 if Path(p).is_file() and Path(p).suffix.lower() in [".txt", ".md"]]
    print(f"[ingest] ROOT: {ROOT}")
    print(f"[ingest] DATA_BASE: {DATA_BASE}")
    print(f"[ingest] найдено PDF: {len(pdf_paths)}, TXT/MD: {len(txt_paths)}")
    for p in pdf_paths:
        docs.extend(PyPDFLoader(p).load())
    for p in txt_paths:
        docs.extend(TextLoader(p, encoding="utf-8").load())
    return docs

def main():
    raw_docs = load_docs()
    if not raw_docs:
        print("[ingest] 0 документов."); sys.exit(1)

    chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_documents(raw_docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(ROOT / "chroma"),
        collection_name="local_docs"
    )
    print("[ingest] ГОТОВО. Индекс в папке 'chroma'.")

if __name__ == "__main__":
    main()
#