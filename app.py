import os
import json
import hashlib
import fitz  # PyMuPDF for reading PDFs
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables (for OpenAI API key)
load_dotenv()

# Ensure OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set your OpenAI API key in the environment variables.")
else:
    openai_api_key = os.getenv("OPENAI_API_KEY")

# Helper function for user authentication
def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password, user_db):
    """Authenticates a user by comparing username and hashed password."""
    if username in user_db and user_db[username] == hash_password(password):
        return True
    return False

def register_user(username, password, user_db):
    """Registers a new user by adding them to the user database."""
    if username in user_db:
        return False  # User already exists
    user_db[username] = hash_password(password)
    with open("user_db.json", "w") as f:
        json.dump(user_db, f)
    return True

# Load or initialize user database
if not os.path.exists("user_db.json"):
    with open("user_db.json", "w") as f:
        json.dump({}, f)

with open("user_db.json", "r") as f:
    user_db = json.load(f)

# Function to process and embed PDF
def get_vectorstore_from_pdf(pdf_file_path):
    """Processes a PDF file and creates a vectorstore using Chroma."""
    # Parse PDF into chunks
    pdf = fitz.open(pdf_file_path)
    text_chunks = []
    for page in pdf:
        text = page.get_text("text")
        text_chunks.extend(text.split("\n\n"))
    pdf.close()

    # Split chunks into smaller pieces
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.create_documents(text_chunks)

    # Create vectorstore with embeddings
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store):
    """Creates a retriever chain for context-aware retrieval."""
    llm = ChatOpenAI(model="gpt-4o")
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation."),
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    """Creates a conversational RAG (retrieval-augmented generation) chain."""
    llm = ChatOpenAI(model="gpt-4o")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    """Gets a response from the RAG chain."""
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input,
    })
    return response["answer"]

# Streamlit UI for login and registration
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("Login / Register")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if authenticate(username, password, user_db):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_username = st.text_input("New Username", key="register_username")
        new_password = st.text_input("New Password", type="password", key="register_password")
        if st.button("Register"):
            if register_user(new_username, new_password, user_db):
                st.success("Registration successful! Please log in.")
            else:
                st.error("Username already exists.")
else:
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state.pop("username", None)
        st.session_state.pop("chat_history", None)
        st.experimental_rerun()

    # PDF Chat Application
    st.title("Chat with your PDF")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! How can I help you?")]

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file:
        # Save uploaded PDF to disk
        file_path = os.path.join("uploads", uploaded_file.name)
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_pdf(file_path)

        # User query
        user_query = st.chat_input("Type your message here...")
        if user_query:
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
