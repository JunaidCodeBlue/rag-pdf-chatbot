import streamlit as st
import fitz  # PyMuPDF
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.schema import Document
import tempfile

# HuggingFace LLM Setup (Free tier)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Title
st.title("📄 RAG Chat with PDF + Memory (Free & Fast)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Store memory in session state
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

if uploaded_file:
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Read and chunk PDF
    doc = fitz.open(tmp_path)
    text = "\n\n".join(page.get_text() for page in doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents([Document(page_content=text)])

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embedding)

    # Setup RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", k=3),
        memory=st.session_state.chat_memory,
    )

    # Chat UI
    st.subheader("Ask a question about the PDF:")
    query = st.text_input("Type your question")

    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
            st.write("### 💡 Answer")
            st.write(answer)

    # Show history
    if st.button("Show Conversation History"):
        st.write(st.session_state.chat_memory.load_memory_variables({})["chat_history"])
