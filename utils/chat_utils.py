import uuid
import datetime
import streamlit as st
from utils.embed_utils import get_embeddings_from_ollama

def create_new_chat():
    """
    Create a new chat session but only store it in session state initially.
    Will be saved to database only when the first message is sent.
    """
    chat_id = str(uuid.uuid4())
    chat_title = f"New Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Store in session state only
    st.session_state.chat_titles[chat_id] = chat_title
    st.session_state.chat_sessions[chat_id] = []
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    
    return chat_id

def load_chat(chat_id, supabase=None):
    """
    Load a chat session from Supabase or session state
    """
    # Reset current messages
    st.session_state.messages = []
    
    # If chat exists in session state with actual messages (not None), load it
    if chat_id in st.session_state.chat_sessions and st.session_state.chat_sessions[chat_id] is not None:
        st.session_state.messages = st.session_state.chat_sessions[chat_id]
        st.session_state.current_chat_id = chat_id
        return
    
    # If not in session state or messages are None but Supabase is available, try to load from database
    if supabase:
        try:
            # Get chat metadata
            chat_response = supabase.table("chat_sessions").select("title").eq("id", chat_id).execute()
            
            if len(chat_response.data) > 0:
                # Store chat title
                st.session_state.chat_titles[chat_id] = chat_response.data[0]["title"]
                
                # Get messages for this chat
                messages_response = supabase.table("chat_messages").select("*").eq("chat_id", chat_id).order("created_at").execute()
                
                if messages_response.data:
                    # Format messages for session state
                    formatted_messages = []
                    for msg in messages_response.data:
                        formatted_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    # Update session state
                    st.session_state.messages = formatted_messages
                    st.session_state.chat_sessions[chat_id] = formatted_messages
                    st.sidebar.success(f"Loaded {len(formatted_messages)} messages from chat history.")
                else:
                    # No messages found, initialize empty list
                    st.session_state.chat_sessions[chat_id] = []
                    st.session_state.messages = []
                    
                st.session_state.current_chat_id = chat_id
            else:
                # Chat not found in database, create a new one
                create_new_chat()
                
        except Exception as e:
            st.error(f"Failed to load chat from database: {str(e)}")
            create_new_chat()
    else:
        # Supabase not available, create a new chat in memory only
        create_new_chat()

def save_message(chat_id, role, content, supabase=None):
    """
    Save a message to the current chat session.
    Also creates the chat session in database if this is the first message.
    
    Args:
        chat_id (str): UUID of the chat session
        role (str): Role of the message sender (user or assistant)
        content (str): Content of the message
        supabase: Supabase client
    """
    # Add to session state first
    message = {"role": role, "content": content}
    
    if chat_id in st.session_state.chat_sessions:
        # Initialize as empty list if it's None
        if st.session_state.chat_sessions[chat_id] is None:
            st.session_state.chat_sessions[chat_id] = []
        st.session_state.chat_sessions[chat_id].append(message)
    else:
        st.session_state.chat_sessions[chat_id] = [message]
    
    # Save to Supabase if available
    if supabase:
        try:
            # Check if this is the first message for this chat
            # If so, create the chat session in the database first
            is_first_message = len(st.session_state.chat_sessions[chat_id]) == 1
            
            if is_first_message and role == "user":
                created_at = datetime.datetime.now().isoformat()
                chat_title = st.session_state.chat_titles.get(chat_id, f"Chat {created_at}")
                
                # Create the chat session in the database
                supabase.table("chat_sessions").insert({
                    "id": chat_id,
                    "title": chat_title,
                    "created_at": created_at,
                }).execute()
            
            # Then save the message
            message_id = str(uuid.uuid4())
            created_at = datetime.datetime.now().isoformat()
            
            supabase.table("chat_messages").insert({
                "id": message_id,
                "chat_id": chat_id,
                "role": role,
                "content": content,
                "created_at": created_at,
            }).execute()
        except Exception as e:
            st.sidebar.error(f"Failed to save message to database: {str(e)}")

def delete_chat(chat_id, supabase=None):
    """
    Delete a chat session from Supabase and session state
    """
    # Remove from session state
    if chat_id in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[chat_id]
    
    if chat_id in st.session_state.chat_titles:
        del st.session_state.chat_titles[chat_id]
    
    # Delete from Supabase if available
    if supabase:
        try:
            # Delete messages first (foreign key constraint)
            supabase.table("chat_messages").delete().eq("chat_id", chat_id).execute()
            
            # Then delete the chat session
            supabase.table("chat_sessions").delete().eq("id", chat_id).execute()
        except Exception as e:
            st.sidebar.error(f"Failed to delete chat from database: {str(e)}")
    
    # If the deleted chat was the current chat, create a new one
    if st.session_state.current_chat_id == chat_id:
        create_new_chat()

def rename_chat(chat_id, new_title, supabase=None):
    """
    Rename a chat session
    """
    # Update in session state
    st.session_state.chat_titles[chat_id] = new_title
    
    # Update in Supabase if available
    if supabase:
        try:
            supabase.table("chat_sessions").update({"title": new_title}).eq("id", chat_id).execute()
        except Exception as e:
            st.sidebar.error(f"Failed to rename chat in database: {str(e)}")

def load_all_chats(supabase):
    """
    Load all chat sessions from Supabase
    """
    if not supabase:
        return
        
    try:
        # Get all chat sessions
        response = supabase.table("chat_sessions").select("id, title, created_at").order("created_at", desc=True).execute()
        
        if response.data:
            for chat in response.data:
                chat_id = chat["id"]
                # Store chat title in session state
                st.session_state.chat_titles[chat_id] = chat["title"]
                
                # We'll load the actual messages only when the chat is selected
                if chat_id not in st.session_state.chat_sessions:
                    st.session_state.chat_sessions[chat_id] = None
    except Exception as e:
        st.sidebar.error(f"Failed to load chats from database: {str(e)}")

def search_documents(chat_id, query, supabase, top_k=5):
    """
    Search for relevant document chunks for a given query
    
    Args:
        chat_id (str): UUID of the chat session
        query (str): User query to find relevant context for
        supabase: Supabase client
        top_k (int): Number of top results to return
        
    Returns:
        list: List of relevant document chunks
    """
    if not supabase:
        return []
    
    try:
        # Get embedding for query using Ollama
        query_embedding = get_embeddings_from_ollama([query], model="nomic-embed-text")[0]
        
        # Run similarity search against user_documents table
        response = supabase.rpc(
            "match_documents", 
            {
                "query_embedding": query_embedding,
                "match_chat_id": chat_id, 
                "match_threshold": 0.5, 
                "match_count": top_k
            }
        ).execute()
        
        if response.data:
            return response.data
        return []
    except Exception as e:
        st.warning(f"Error searching documents: {str(e)}")
        return []