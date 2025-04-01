import io
import re
import uuid
import datetime
import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_uploaded_file(uploaded_file):
    """
    Extract text from an uploaded file (PDF, DOC, DOCX)
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Extracted text from the document
    """
    file_name = uploaded_file.name.lower()
    file_content = uploaded_file.read()
    
    try:
        if file_name.endswith('.pdf'):
            # Extract text from PDF
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
                
        elif file_name.endswith('.docx'):
            # Extract text from DOCX
            docx_file = io.BytesIO(file_content)
            text = docx2txt.process(docx_file)
            
        elif file_name.endswith('.doc'):
            # For DOC files, this is a simplified approach
            # A more robust solution might use other libraries
            text = str(file_content, 'latin-1', errors='ignore')
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
            
        else:
            text = "Unsupported file format. Please upload a PDF, DOC, or DOCX file."
            
        return text.strip()
    except Exception as e:
        return f"Error extracting text from file: {str(e)}"

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Split text into smaller chunks for processing
    
    Args:
        text (str): Text to split into chunks
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters
        
    Returns:
        list: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.create_documents([text])
    return [chunk.page_content for chunk in chunks]

def save_document_chunks_to_supabase(supabase, chat_id, file_name, chunks, embeddings):
    """
    Save document chunks and their embeddings to Supabase
    
    Args:
        supabase: Supabase client
        chat_id (str): UUID of the chat session
        file_name (str): Name of the uploaded file
        chunks (list): List of text chunks
        embeddings (list): List of embeddings vectors
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not supabase:
        return False
    try:
        # Create a batch of records to insert
        records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            created_at = datetime.datetime.now().isoformat()
            
            records.append({
                "id": chunk_id,
                "chat_id": chat_id,
                "file_name": file_name,
                "chunk_index": i,
                "content": chunk,
                "embedding": embedding,
                "created_at": created_at,
            })
            
        # Insert chunks in batches to avoid payload size issues
        batch_size = 10
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase.table("user_documents").insert(batch).execute()
            
        return True
        
    except Exception as e:
        st.sidebar.error(f"Failed to save document chunks: {str(e)}")
        return False