import streamlit as st
from retriever import get_retriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def LegalChatbot():
    return Ollama(model="llama3.2:3b")

llm = LegalChatbot()
retriever = get_retriever()

template = """
You are a legal expert assistant. Use the context to answer user questions.

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

st.title("📜 Legal Document Assistant (LLM + Ollama)")

user_input = st.text_input("Ask a legal question:")
if user_input:
    result = qa_chain.invoke({"query": user_input})
    st.write("### Answer:")
    st.write(result["result"])
    st.write("### Sources:")
    for doc in result["source_documents"]:
        st.markdown(f"**{doc.metadata.get('filename')}**, Page {doc.metadata.get('page_number')}")
        st.write(doc.page_content[:200])