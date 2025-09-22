import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Import the necessary LLM classes
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from a .env file
load_dotenv()

# --- UI elements and configuration in the sidebar ---
st.title("📚 RAG App with PDF & Web Support")
st.markdown("Ask questions based on your documents. Powered by LangChain, Google Gemini, and Ollama!")

# LLM Selection and configuration in the sidebar
with st.sidebar:
    st.header("⚙️ App Settings")

    # Data Source Selection
    data_source = st.radio(
        "Choose your data source:",
        ("PDF File", "Website URL")
    )

    # File or URL input based on selection
    uploaded_file = None
    url_input = ""
    if data_source == "PDF File":
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    else:
        url_input = st.text_input("Enter a website URL:")
    
    # LLM Provider Selection
    llm_provider = st.selectbox(
        "Select LLM Provider", 
        ("Ollama", "Google Gemini"),
        help="Choose between a local Ollama model or a cloud-based Google Gemini model."
    )
    
    # Dynamic Model Selection and API Key input based on provider
    if llm_provider == "Ollama":
        llm_model = st.selectbox(
            "Select Ollama Model", 
            ("phi3:mini", "gemma:2b", "neural-chat", "orca-mini"),
            help="Make sure the selected model is running in your Ollama instance."
        )
        api_key = None
    elif llm_provider == "Google Gemini":
        llm_model = st.selectbox(
            "Select Google Gemini Model",
            ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest"),
            help="These models are available through the Google Generative AI API."
        )
        api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            api_key = st.text_input(
                "Enter your Google API Key",
                type="password",
                help="You can get your API key from https://aistudio.google.com/app/apikey"
            )

    # Chunking Parameters
    st.subheader("Text Splitter Settings")
    chunk_size = st.slider("Chunk Size", 100, 2000, 1000, 100, help="The maximum number of characters in a chunk.")
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50, help="The number of overlapping characters between chunks.")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# --- Session State Initialization and Data Processing ---
config_key = f"{uploaded_file.name if uploaded_file else url_input}-{llm_provider}-{llm_model}-{chunk_size}-{chunk_overlap}"
if "qa_chain" not in st.session_state or st.session_state.get("config_key") != config_key:
    st.session_state.qa_chain = None
    st.session_state.config_key = config_key
    st.session_state.processed = False

# Setup directory for saving vectorstores
VECTOR_DIR = "vectorstores"
os.makedirs(VECTOR_DIR, exist_ok=True)

# Process PDF or URL and create/load vectorstore
if (uploaded_file or url_input) and not st.session_state.processed:
    # Replaced st.spinner with st.status for more granular control
    with st.status("🔍 Processing document...", expanded=True) as status:
        docs = []
        if data_source == "PDF File" and uploaded_file:
            status.update(label="Downloading and saving PDF...", state="running")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                pdf_path = temp_file.name
            
            status.update(label="Loading PDF content...", state="running")
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
            os.remove(pdf_path)

        elif data_source == "Website URL" and url_input:
            try:
                status.update(label="Loading website content...", state="running")
                loader = WebBaseLoader(url_input)
                docs.extend(loader.load())
            except Exception as e:
                status.update(label=f"Error loading URL: {e}", state="error")
                st.error(f"Error loading URL: {e}")
                st.session_state.processed = False
                st.stop()
        
        status.update(label="Splitting text into chunks...", state="running")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        docs = text_splitter.split_documents(docs)

        status.update(label="Creating embeddings...", state="running")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if uploaded_file:
            file_id = os.path.splitext(uploaded_file.name)[0]
        else:
            sanitized_url = re.sub(r'[\\/:*?"<>|]', '_', url_input)
            file_id = sanitized_url
        
        vector_path = os.path.join(VECTOR_DIR, file_id)

        if os.path.exists(vector_path):
            status.update(label="Loading existing vector store...", state="running")
            vectordb = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
        else:
            status.update(label="Generating vector store...", state="running")
            vectordb = FAISS.from_documents(docs, embeddings)
            vectordb.save_local(vector_path)

        status.update(label="Initializing LLM...", state="running")
        llm = None
        if llm_provider == "Ollama":
            llm = Ollama(model=llm_model, temperature=0)
        elif llm_provider == "Google Gemini":
            if not api_key:
                status.update(label="Please enter your Google API Key to use Gemini.", state="error")
                st.error("Please enter your Google API Key to use Gemini.")
                st.session_state.processed = False
                st.stop()
            llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0, api_key=api_key, streaming=True)
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(),
            memory=memory
        )

        st.session_state.qa_chain = qa_chain
        st.session_state.processed = True
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

    # Final status update
    status.update(label="✅ Document Processed! You can now ask your question below.", state="complete", expanded=False)

# Display chat messages from history on app rerun
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your question:"):
    if not st.session_state.qa_chain:
        st.warning("Please upload or enter a document first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in st.session_state.qa_chain.stream({"question": prompt}):
            if "answer" in chunk:
                full_response += chunk["answer"]
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

st.markdown("---")
st.markdown("Made with ❤️ using [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), and LLMs from Ollama and Google.")
