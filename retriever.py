import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from typing import List
import logging
from langchain.schema import Document

model = "llama3.2:3b"

PDF_DIR = "./data"
DB_DIR = "./embeddings"
EMBEDDING_MODEL = "nomic-embed-text:latest"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(pdf_dir: str) -> List[Document]:
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF Directory '{pdf_dir}' does not exist.")

    all_docs = []
    for file_name in os.listdir(pdf_dir):
        if file_name.endswith(".pdf"):
            try:
                file_path = os.path.join(pdf_dir, file_name)
                logger.info(f"Loading document: {file_path}")
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                for i, doc in enumerate(docs):
                    if not doc.page_content.strip():
                        logger.warning(f"Skipping empty page {i+1} in {file_name}")
                        continue
                        
                    new_doc = Document(
                        page_content=doc.page_content,
                        metadata={
                            "file_name": file_name,
                            "source": file_name,
                            "filepath": file_path,
                            "page_number": i + 1
                        }
                    )
                    all_docs.append(new_doc)

                logger.info(f"Loaded {len(all_docs)} pages from {file_name}")
            except Exception as e:
                logger.error(f"Error loading {file_name}: {str(e)}")
                continue

    return all_docs

def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap = 50
    )   

    return splitter.split_documents(documents)

def create_vector_store(chunks: List[Document], persist_directory: str = DB_DIR) -> Chroma:
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_name = "Legal_Documents"
    )

    return vectordb

def get_retriever():
    try:
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

        if os.path.exists(DB_DIR):
            logger.info(f"Loading existing vector store:")
            vectordb = Chroma(
                persist_directory = DB_DIR,
                embedding_function = embedding_model
            )
        else:
            logger.info(f"Creating new vector store:")
            os.makedirs(DB_DIR, exist_ok = True)
            raw_docs = load_documents(PDF_DIR)
            if not raw_docs:
                raise ValueError("No documents found in the specified directory.")
            chunks = split_documents(raw_docs)
            vectordb = create_vector_store(chunks)

        return vectordb.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        logger.warning(f"Error initializing retriever: {str(e)}")
        raise RuntimeError(f"Failed to initialize retriever: {str(e)}")