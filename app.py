import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize session
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# UI
st.title("📚 PDF Q&A Tool with Mistral (via Ollama)")
st.markdown("Ask questions based on your PDFs. Powered by LangChain + Ollama + Mistral!")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Setup directory for saving vectorstores
VECTOR_DIR = "vectorstores"
os.makedirs(VECTOR_DIR, exist_ok=True)

# Process PDF or load existing vectorstore
if uploaded_file is not None and not st.session_state.processed:
    with st.spinner("🔍 Processing PDF..."):
        # Save uploaded file to disk
        pdf_path = os.path.join("temp_pdf.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Use filename as unique ID
        file_id = os.path.splitext(uploaded_file.name)[0]
        vector_path = os.path.join(VECTOR_DIR, file_id)

        # Try to load vectorstore from disk
        if os.path.exists(vector_path):
            vectordb = FAISS.load_local(vector_path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
        else:
            # Load + split PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(pages)

            # Create vectorstore
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectordb = FAISS.from_documents(docs, embeddings)

            # Save vectorstore
            vectordb.save_local(vector_path)

        # Load LLM via Ollama
        llm = Ollama(model="mistral:7b", temperature=0)

        # QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )

        # Cache in session
        st.session_state.qa_chain = qa_chain
        st.session_state.processed = True

    st.success("✅ PDF Processed! You can now ask your question below.")

# Ask question
if st.session_state.qa_chain:
    question = st.text_input("🧠 Enter your question:")

    if question:
        with st.spinner("💬 Generating answer with Mistral..."):
            result = st.session_state.qa_chain({"query": question})
            answer = result["result"]
            sources = result.get("source_documents", [])

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for doc in sources:
            st.write(f"📄 {doc.metadata.get('source', 'Unknown')}")
            st.write(doc.page_content[:300] + "...")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), and [Ollama](https://ollama.com)")
