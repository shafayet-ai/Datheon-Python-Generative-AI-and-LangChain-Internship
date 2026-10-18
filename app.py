import os
import tempfile
import streamlit as st # pyright: ignore[reportMissingImports]

# Document processing & vector store
from langchain_community.document_loaders import PyPDFLoader  # pyright: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import FAISS  # pyright: ignore[reportMissingImports]
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # pyright: ignore[reportMissingImports]

# LangChain classic chains (for compatibility with LangChain 1.0+)
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain  # pyright: ignore[reportMissingImports]
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # pyright: ignore[reportMissingImports]

# Prompts & messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage, AIMessage  # pyright: ignore[reportMissingImports]


st.set_page_config(
    page_title="Interactive RAG AI Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


def process_documents(uploaded_files, openai_api_key):
    documents = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        documents.extend(docs)
        os.remove(tmp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def build_rag_chain(vectorstore, openai_api_key, model_name, temperature):
   
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key,
        streaming=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an intelligent assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


with st.sidebar:
    st.title("⚙️ RAG Settings")
    
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Type your OpenAI API key here. You can get it from https://platform.openai.com/account/api-keys"
    )
    
    st.divider()
    
    selected_model = st.selectbox(
        "Select Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo","gpt-5.6", "gpt-5.6-mini", "gpt-5.6-large"],
        index=0
    )
    temp_slider = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    
    st.subheader(" Upload Knowledge Base")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(" Process Documents", use_container_width=True):
        if not api_key_input:
            st.error("please provide your OpenAI API Key in the sidebar!")
        elif not uploaded_pdfs:
            st.warning("please upload at least one PDF file!")
        else:
            with st.spinner("processing data and indexing vectors..."):
                try:
                    st.session_state.vectorstore = process_documents(uploaded_pdfs, api_key_input)
                    st.session_state.rag_chain = build_rag_chain(
                        st.session_state.vectorstore,
                        api_key_input,
                        selected_model,
                        temp_slider
                    )
                    st.success("documents processed and indexed successfully!")
                except Exception as e:
                    st.error(f"Error occurred: {str(e)}")

    st.divider()
    
    if st.button(" Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.markdown('<div class="chat-header"> Interactive RAG AI Chatbot</div>', unsafe_allow_html=True)
st.caption("Streamlit, LangChain, and OpenAI using RAG to create an interactive chatbot.")

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if user_input := st.chat_input("Ask a question about your documents..."):
    
    if not api_key_input:
        st.info("⚠️ Please provide your OpenAI API Key in the sidebar.")
        st.stop()

    if st.session_state.rag_chain is None:
        st.info(" No PDF documents indexed. Answer will be generated directly by the AI model.")

    # Display user query in UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Format LangChain chat history
    langchain_history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            langchain_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            langchain_history.append(AIMessage(content=m["content"]))

    # Stream Response from Assistant
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if st.session_state.rag_chain:
            try:
                response_stream = st.session_state.rag_chain.stream({
                    "input": user_input,
                    "chat_history": langchain_history
                })

                for chunk in response_stream:
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
        else:
            llm = ChatOpenAI(
                model=selected_model,
                temperature=temp_slider,
                openai_api_key=api_key_input,
                streaming=True
            )
            for chunk in llm.stream(user_input):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})