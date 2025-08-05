# 📚 PDF Q&A Tool with Mistral (via Ollama)

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?style=flat-square&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-enabled-blue?style=flat-square)
![Mistral 7B](https://img.shields.io/badge/Model-Mistral%207B-orange?style=flat-square)
![Ollama](https://img.shields.io/badge/Backend-Ollama-4B3263?style=flat-square)

A local, privacy-focused search engine that allows you to ask **natural language questions about your PDF documents**. Powered by **LangChain**, **Ollama**, and **Streamlit**, this app provides answers with source references from your files — all processed **entirely on your machine** using open-source tools.

---

## ✨ Features

- 📄 **PDF Upload & Processing**: Load PDFs and split them into smart chunks for better context understanding.
- 🧠 **Local LLM**: Uses **Mistral 7B** via [Ollama](https://ollama.com) for offline, private inference.
- 🗂️ **FAISS Vector Store**: Stores embeddings locally using [FAISS](https://github.com/facebookresearch/faiss) for fast and persistent similarity search.
- 💬 **Contextual Question Answering**: Retrieves the most relevant chunks to answer your questions.
- 🧾 **Source Citations**: See the exact document excerpts used to generate each answer.
- 🧠 **Session Caching**: Avoids reprocessing PDFs during the same session using `st.session_state`.

---

## 🛠 Tech Stack

| Component         | Tech / Library |
|------------------|----------------|
| UI               | Streamlit      |
| LLM              | Mistral 7B (via Ollama) |
| Vector Store     | FAISS          |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` |
| Document Loader  | LangChain's `PyPDFLoader` |
| QA Chain         | LangChain's `RetrievalQA` |

---

## 📦 Installation

### ✅ Prerequisites

- Install [Ollama](https://ollama.com) and ensure it is running:
  ```bash
  ollama run mistral
  ```

- Download the required model:
  ```bash
  ollama pull mistral:7b
  ```

### 🔧 Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/pdf-qa-tool.git
   cd pdf-qa-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. Start the app:
   ```bash
   streamlit run app.py
   ```

2. Upload a PDF file using the file uploader.

3. Ask any question related to the content of the PDF.

4. View the generated answer and its source references.

---

## 🤖 Powered By

- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)

---

## 🔐 Privacy First

> All processing is done **locally on your machine**. Your documents and questions are never sent to any external server.
