#  Interactive Full-Stack AI Chatbot with Streamlit

A responsive, production-ready full-stack AI web application featuring real-time token streaming, document ingestion via RAG (Retrieval-Augmented Generation), vector persistence using FAISS, and conversation memory management.

---

##  Key Features

* **Interactive Streamlit Web UI:** Built using native Streamlit chat components (`st.chat_message`, `st.chat_input`).
* **Real-time Token Streaming:** Displays responses token-by-token for a seamless user experience.
* **Document Ingestion (RAG):** Upload multi-page PDF documents to process, split, and index into a vector store.
* **Conversation History & Session State:** Persists full chat context across reruns using `st.session_state`.
* **Multi-Backend Support:** Works with OpenAI models as well as local open-source LLMs via Ollama.

---

## 📁 Project Structure

```text
├── app.py              # Main Streamlit application
├── README.md    # Documentation
└── requirements.txt         # Python dependencies 
```

---

##  Quick Start & Installation

### 1. Prerequisites

Ensure you have **Python 3.10+** installed on your system.

### 2. Clone/Setup Project Directory

Open your terminal and navigate to your project directory:
```bash
cd "Interactive Full-Stack AI Chatbot"
```

### 3. Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install streamlit langchain langchain-classic langchain-community langchain-openai pypdf faiss-cpu python-dotenv
```

*(Optional: If running fully local models without OpenAI credits, install `langchain-ollama`)*
```bash
pip install langchain-ollama
```

---

## 💻 Running the Application

To launch the web interface, execute:

```bash
streamlit run app.py
```

Once executed, open your web browser at:
`http://localhost:8501`

---

## ⚙️ How to Use

1. **Enter API Key / Model:**
   * Enter your **OpenAI API Key** in the sidebar.
   * Alternatively, use **Ollama** for free local model execution.
2. **Upload Documents:**
   * Drag and drop single or multiple `.pdf` files into the knowledge base section.
   * Click **Process Documents** to chunk, embed, and store document context into the FAISS vector database.
3. **Start Chatting:**
   * Type any question in the bottom chat bar.
   * Watch the AI stream answers token-by-token based on your uploaded document context!