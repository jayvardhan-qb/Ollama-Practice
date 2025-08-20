from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from retriever import get_retriever
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

def LegalChatbot():
    return Ollama(model=LLM_MODEL, temperature = 0.4)

llm = LegalChatbot()
retriever = get_retriever()

template = """
You are a legal assistant trained on Indian law documents. Respond clearly with the provided context.
If unsure, say "I'm not sure, consult a legal expert." 
Keep in mind to respond only to context related questions, do not answer for out of the context questions. 
End with: "This is a legal information service, not a substitute for legal advice."

Context: {context}

Question: {question}
"""

def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant documents found"
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate(
    input_variables = ["context", "question"],
    template = template
)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# qa_chain = RetrievalQA.from_chain_type(
#     llm = llm,
#     retriever = retriever,
#     chain_type = "stuff",
#     chain_type_kwargs = {"prompt": prompt},
#     return_source_documents = True,
#     output_key = "result"
# )

class DocumentSource(BaseModel):
    file_name: str
    page_number: int
    source_path: str

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[DocumentSource]

class SummaryResponse(BaseModel):
    summary: str
    file_name: str
    total_pages: int

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    try:
        docs = retriever.invoke(req.question)
        context = format_docs(docs)
        answer = llm.invoke(
            prompt.format(context = context, question = req.question)
        )
        sources = [
            DocumentSource(
                file_name = doc.metadata.get("file_name", "Unknown"),
                page_number = doc.metadata.get("page_number", 0),
                source_path = os.path.basename(doc.metadata.get("source", "Unknown"))
            # {
            #     "file_name": doc.metadata.get("file_name", "Unknown"),
            #     "page_number": doc.metadata.get("page_number", 0),
            #     "source_path": os.path.basename(doc.metadata.get("source", "Unknown"))
            # }
            )
            for doc in docs
        ]

        return AnswerResponse(
            answer = answer,
            sources = sources
        )
        # result = qa_chain.invoke({"query": req.question})
        # return {
        #     "answers": result["result"]
        # }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}

@app.post("/summarize", response_model=SummaryResponse)
def summarize_documents(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_file_path)
            raw_docs = loader.load()
            
            docs = []
            for i, doc in enumerate(raw_docs):
                if not doc.page_content.strip():
                    continue
                docs.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        "file_name": file.filename,
                        "source": file.filename,
                        "filepath": temp_file_path,
                        "page_number": i + 1
                    }
                ))

            if not docs:
                raise HTTPException(status_code=400, detail="No readable content found in PDF")

            embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vector_db = Chroma(
                persist_directory='./embeddings',
                embedding_function=embedding_model
            )
            vector_db.add_documents(docs)

            full_text = ".\n".join([doc.page_content for doc in docs])

            summary_prompt = f"""
            Provide a detailed summary by covering:
            1. Key points and arguments
            2. Important clauses and terms
            3. Top 3 risks or unusual clauses
            4. Governing law and jurisdiction

            \\n\\n{full_text[:8000]}
            Make it concise and easy to understand.
            """

            summary = llm.invoke(summary_prompt)

            return SummaryResponse(
                summary= summary,
                file_name= file.filename,
                total_pages= len(docs)
            )
        finally:
            os.unlink(temp_file_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        raise HTTPException(500, "Error summarizing the document")

# @app.post("/summarize")
# def summarize_documents(file: UploadFile = File(...)):
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(file.file.read())
#             temp_file_path = temp_file.name

#         try:
#             loader = PyPDFLoader(temp_file_path)
#             docs = loader.load()

#             for i, doc in enumerate(docs):
#                 doc.metadata.update({
#                     "filename": file.filename,
#                     "filepath": temp_file_path,
#                     "page_number": i + 1
#                 })

#             embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

#             vector_db = Chroma(
#                 persist_directory = './embeddings',
#                 embedding_function = embedding_model
#             )

#             vector_db.add_documents(docs)

#             full_text = ".\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
#             if not full_text:
#                 raise HTTPException(400, "No text found in the document.")

#             summary_prompt = f"""
#             Provide a detailed summary by covering:
#             1. Key points and arguments
#             2. Important clauses and terms
#             3. Top 3 risks or unusual clauses
#             4. Governing law and jurisdiction

#             \\n\\n{full_text[:8000]}
#             Make it concise and easy to understand.
#             """

#             summary = llm.invoke(summary_prompt)

#             return {
#                 "summary": summary
#             }
#         finally:
#             os.unlink(temp_file_path)
#     except Exception as e:
#         logger.error(f"Error summarizing document: {str(e)}")
#         raise HTTPException(
#             status_code = 500,
#             detail = "Error summarizing the document"
#         )