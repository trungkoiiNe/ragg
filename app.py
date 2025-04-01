import streamlit as st
import os
from dotenv import load_dotenv

# Import utility modules
from utils.db_utils import initialize_supabase_client, initialize_database
from utils.doc_utils import extract_text_from_uploaded_file, chunk_text, save_document_chunks_to_supabase
from utils.embed_utils import get_embeddings_from_ollama, check_ollama_server
from utils.chat_utils import (
    create_new_chat, load_chat, save_message, delete_chat, 
    rename_chat, load_all_chats, search_documents
)
from utils.llm_utils import get_llm_response

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize Supabase client
supabase = initialize_supabase_client()

# Set page configuration
st.set_page_config(page_title="RAG4ALL", page_icon="🤖", layout="wide")

# Initialize session state variables
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}
if "show_document_details" not in st.session_state:
    st.session_state.show_document_details = False
    
# Ensure messages is always a list to prevent TypeError when iterating
if st.session_state.messages is None:
    st.session_state.messages = []

# Initialize database and load chats
if supabase:
    db_initialized = initialize_database(supabase)
    if db_initialized:
        load_all_chats(supabase)

# If no chat is selected or available, create a new one
if not st.session_state.current_chat_id or st.session_state.current_chat_id not in st.session_state.chat_sessions:
    create_new_chat()

# Main application header
st.header("RAG4ALL Chat")

# Sidebar configuration
with st.sidebar:
    st.title("RAG4ALL")
    st.markdown("---")
    
    # Chat sessions management
    st.subheader("Chat Sessions")
    
    # Button to create a new chat
    if st.button("New Chat", key="new_chat"):
        create_new_chat()
        st.rerun()
    
    # Display and select available chats
    chat_ids = list(st.session_state.chat_titles.keys())
    
    if chat_ids:
        for chat_id in chat_ids:
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            
            # Add button behavior to chat title
            with col1:
                if st.button(st.session_state.chat_titles[chat_id], key=f"chat_{chat_id}", use_container_width=True):
                    load_chat(chat_id, supabase)
                    st.rerun()
            
            # Edit button for renaming chat
            with col2:
                if st.button("✏️", key=f"edit_{chat_id}"):
                    st.session_state.editing_chat_id = chat_id
            
            # Delete button
            with col3:
                if st.button("🗑️", key=f"delete_{chat_id}"):
                    delete_chat(chat_id, supabase)
                    st.rerun()
                    
        # Rename chat popup
        if "editing_chat_id" in st.session_state:
            chat_id = st.session_state.editing_chat_id
            if chat_id in st.session_state.chat_titles:
                current_title = st.session_state.chat_titles[chat_id]
                new_title = st.text_input("New title", value=current_title, key=f"rename_{chat_id}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save", key=f"save_{chat_id}"):
                        rename_chat(chat_id, new_title, supabase)
                        del st.session_state.editing_chat_id
                        st.rerun()
                with col2:
                    if st.button("Cancel", key=f"cancel_{chat_id}"):
                        del st.session_state.editing_chat_id
                        st.rerun()
    
    # Model and API settings
    st.markdown("---")
    st.subheader("Settings")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                              help="Controls randomness: 0 is deterministic, 1 is more creative")
        max_tokens = st.slider("Max Tokens", min_value=100, max_value=4000, value=1000, step=100,
                             help="Maximum number of tokens to generate")
    
    # Model selection
    model = st.selectbox(
        "Model", 
        ["meta-llama/llama-3-8b-instruct:free"], 
        index=0,
        help="The AI model used for generating responses"
    )
    
    # Display API status
    st.markdown("---")
    st.subheader("API Status")
    
    # OpenRouter API status
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_api_key_here":
        st.success("OpenRouter API Key: ✅")
    else:
        st.error("OpenRouter API Key: ❌")
        st.info("Update the OPENROUTER_API_KEY value in the .env file and restart the app.")
    
    # Supabase API status
    if supabase:
        st.success("Supabase Connection: ✅")
    else:
        st.warning("Supabase Connection: ❌")
        st.info("Update the SUPABASE_URL and SUPABASE_KEY values in the .env file for database functionality.")
    
    st.markdown("---")
    st.subheader("About")
    st.markdown("""\
    **RAG4ALL** is a Streamlit chat application featuring:
    
    - Multiple independent chat sessions
    - Persistent conversations with Supabase
    - Integration with OpenRouter LLM API
    
    RAG stands for Retrieval-Augmented Generation, a technique that enhances 
    large language models with additional context from external knowledge sources.
    """)
    
    # Clear current chat button
    if st.button("Clear Current Chat"):
        if st.session_state.current_chat_id:
            st.session_state.messages = []
            st.session_state.chat_sessions[st.session_state.current_chat_id] = []
            
            # Clear messages from database if available
            if supabase:
                try:
                    supabase.table("chat_messages").delete().eq("chat_id", st.session_state.current_chat_id).execute()
                except Exception as e:
                    st.error(f"Failed to clear chat messages from database: {str(e)}")
                    
            st.rerun()

# Main chat area
current_chat_title = "New Chat"
if st.session_state.current_chat_id in st.session_state.chat_titles:
    current_chat_title = st.session_state.chat_titles[st.session_state.current_chat_id]

st.subheader(f"Current Chat: {current_chat_title}")

# Document upload section above the chat interface
with st.expander("📄 Upload Documents for this Chat", expanded=False):
    st.write("Upload PDF, DOCX, or DOC files to enhance the AI's knowledge for this chat session.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        help="Upload files to provide context for your questions"
    )
    
    # Process uploaded files
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents... This may take a while."):
            for uploaded_file in uploaded_files:
                # Extract text from the uploaded file
                st.write(f"Processing {uploaded_file.name}...")
                
                # Extract text
                text = extract_text_from_uploaded_file(uploaded_file)
                
                if text.startswith("Error:"):
                    st.error(text)
                    continue
                
                # Check if Ollama server is running
                if not check_ollama_server():
                    st.error("Ollama server is not running. Make sure it's running on localhost:11434.")
                    st.info("Install Ollama from https://ollama.ai/ and run 'ollama pull nomic-embed-text'")
                    continue
                
                # Chunk the text into smaller pieces for embedding
                chunks = chunk_text(text)
                st.write(f"Created {len(chunks)} chunks from the document.")
                
                # Generate embeddings for the chunks
                with st.spinner(f"Generating embeddings for {uploaded_file.name}..."):
                    embeddings = get_embeddings_from_ollama(chunks)
                
                # Save to Supabase
                if save_document_chunks_to_supabase(
                    supabase,
                    st.session_state.current_chat_id,
                    uploaded_file.name,
                    chunks,
                    embeddings
                ):
                    st.success(f"Successfully processed {uploaded_file.name}")
                else:
                    st.error(f"Failed to save {uploaded_file.name} to the database.")
            
            st.success("All documents processed!")
    
    # Display uploaded documents for this chat
    if st.session_state.current_chat_id and supabase:
        try:
            # Get all documents for current chat
            docs_response = supabase.table("user_documents")\
                .select("file_name")\
                .eq("chat_id", st.session_state.current_chat_id)\
                .execute()
            
            if docs_response.data:
                # Process the document data to count chunks per file
                doc_counts = {}
                for doc in docs_response.data:
                    file_name = doc['file_name']
                    if file_name in doc_counts:
                        doc_counts[file_name] += 1
                    else:
                        doc_counts[file_name] = 1
                
                # Display document information
                if doc_counts:
                    st.subheader("Uploaded Documents")
                    for file_name, count in doc_counts.items():
                        st.write(f"• {file_name} ({count} chunks)")
                    
                    # Option to delete documents
                    if st.button("Delete All Documents for This Chat"):
                        supabase.table("user_documents")\
                            .delete()\
                            .eq("chat_id", st.session_state.current_chat_id)\
                            .execute()
                        st.success("All documents deleted!")
                        st.rerun()
            
        except Exception as e:
            st.error(f"Error retrieving document list: {str(e)}")

# Display chat messages for the current chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask something..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Save user message to database
    save_message(st.session_state.current_chat_id, "user", prompt, supabase)
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_api_key_here":
                response = get_llm_response(prompt, st.session_state.current_chat_id, supabase, temperature, max_tokens)
            else:
                response = "Please set up your OpenRouter API key in the .env file to enable API calls."
        st.write(response)
    
    # Add assistant response to chat history and save to database
    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(st.session_state.current_chat_id, "assistant", response, supabase)