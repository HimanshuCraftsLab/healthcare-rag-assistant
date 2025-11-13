# --- Import Libraries ---
import os  # For environment variables and file operations
import streamlit as st  # For building the web interface
from PyPDF2 import PdfReader  # For reading PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_community.embeddings import OpenAIEmbeddings  # For generating text embeddings
from langchain_community.vectorstores import FAISS  # For vector storage and retrieval
from langchain.chains import RetrievalQA  # For question answering chain
from langchain_openai import OpenAI  # For OpenAI LLM integration
from langchain.prompts import PromptTemplate  # For creating prompt templates
from dotenv import load_dotenv  # For loading environment variables

# --- Load Environment Variables ---
# Load variables from .env file (contains OpenAI API key)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Validate API Key ---
# Check if API key exists, show error and stop if not found
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in `.env` file. Please add it!")
    st.stop()

# --- Streamlit Configuration ---
# Set up the page title and layout
st.set_page_config(page_title="Healthcare RAG Assistant", layout="wide")
st.title("🏥 Healthcare RAG Assistant (OpenAI)")

# --- Application Constants ---
TEMPERATURE = 0.1  # Controls randomness of LLM responses (lower = more deterministic)
PROMPT_TEMPLATE = """
You are a clinical documentation specialist analyzing medical records. 
Generate a professional structured report using ONLY the provided context.
Maintain strict accuracy - do not infer information not present in the context.

CONTEXT:
{context}

QUESTION:
{question}
"""

# --- Document Processing Functions ---
def process_documents(uploaded_files):
    """
    Processes uploaded files (PDF or text) and extracts all text content.
    
    Args:
        uploaded_files: List of files uploaded via Streamlit file_uploader
        
    Returns:
        str: Combined text from all uploaded files
    """
    text = ""
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            # Extract text from each page of PDF
            reader = PdfReader(file)
            text += "\n".join([page.extract_text() for page in reader.pages])
        else:
            # Read text directly from text files
            text += file.read().decode("utf-8")
    return text

def chunk_text(text):
    """
    Splits text into smaller chunks for processing by the language model.
    
    Args:
        text (str): The complete text to be chunked
        
    Returns:
        List[str]: List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Maximum size of each chunk
        chunk_overlap=200,  # Overlap between chunks for context preservation
        separators=["\n\n", "\n", "(?<=\. )", " "]  # Preferred split points
    )
    return splitter.split_text(text)

# --- Core RAG Workflow Functions ---
def setup_rag(text_chunks):
    """
    Sets up the Retrieval-Augmented Generation pipeline.
    
    1. Creates embeddings for text chunks
    2. Builds a vector store for semantic search
    3. Configures the QA retrieval chain
    
    Args:
        text_chunks: List of text chunks to be indexed
        
    Returns:
        RetrievalQA: Configured question answering chain
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Create vector store from text chunks
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    
    # Configure the QA retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=TEMPERATURE, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",  # Simple chain type for stuffing all docs into prompt
        retriever=vector_store.as_retriever(),  # Configure retriever from vector store
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=PROMPT_TEMPLATE,  # Use our custom template
                input_variables=["context", "question"]  # Expected variables
            )
        }
    )
    return qa_chain

# --- Main Application Function ---
def main():
    """
    Main function that runs the Streamlit application.
    Handles file uploads, processing, and question answering.
    """
    # Sidebar file uploader
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF/TXT files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )

    # Process uploaded files if any
    if uploaded_files:
        with st.spinner("Processing documents..."):
            # Extract text from all files
            text = process_documents(uploaded_files)
            # Split text into chunks
            chunks = chunk_text(text)
            # Set up RAG pipeline and store in session state
            st.session_state.qa_chain = setup_rag(chunks)
            st.sidebar.success(f"Processed {len(chunks)} text chunks!")

    # Question input and answer generation
    query = st.text_input("Ask a question (e.g., 'Summarize key findings'):")
    if query and "qa_chain" in st.session_state:
        with st.spinner("Generating response..."):
            # Get answer from QA chain
            response = st.session_state.qa_chain.run(query)
            # Display formatted response
            st.markdown(response)

# --- Entry Point ---
if __name__ == "__main__":
    main()
