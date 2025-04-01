import os
import requests
import datetime
import streamlit as st
from supabase import create_client, Client  # type: ignore

def initialize_supabase_client():
    """Initialize and return a Supabase client using environment variables."""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    supabase = None
    if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL != "your_supabase_url":
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            st.error(f"Failed to connect to Supabase: {str(e)}")
    
    return supabase

def initialize_database(supabase):
    """
    Create necessary tables in Supabase if they don't exist
    """
    if not supabase:
        return False
        
    try:
        # Check if tables exist by trying to query them
        tables_exist = False
        try:
            # Try to get table info from Supabase
            response = supabase.table("chat_sessions").select("id").limit(1).execute()
            tables_exist = True
        except Exception:
            # Tables might not exist
            tables_exist = False
            
        # If tables don't exist, create them via REST API requests
        if not tables_exist:
            try:
                # Create tables directly using Supabase REST API
                # For this to work, make sure you have enabled "Bypass RLS" for the service key
                
                # Get the Supabase key and URL from the client
                SUPABASE_URL = os.getenv("SUPABASE_URL")
                SUPABASE_KEY = os.getenv("SUPABASE_KEY")
                
                # Build the headers for direct REST API calls
                headers = {
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                }
                
                # Create chat_sessions table
                create_sessions_sql = """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id UUID PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL
                );
                """
                
                # Create chat_messages table with foreign key
                create_messages_sql = """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id UUID PRIMARY KEY,
                    chat_id UUID NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );
                """
                
                # Ensure the user_documents table exists with proper vector support
                ensure_vector_extension_and_tables(SUPABASE_URL, SUPABASE_KEY)
                
                # Execute SQL directly via REST API
                requests.post(
                    f"{SUPABASE_URL}/rest/v1/rpc/exec_sql",
                    headers=headers,
                    json={"query": create_sessions_sql}
                ).raise_for_status()
                
                requests.post(
                    f"{SUPABASE_URL}/rest/v1/rpc/exec_sql",
                    headers=headers,
                    json={"query": create_messages_sql}
                ).raise_for_status()
                
                st.sidebar.success("Database tables created successfully!")
                
            except Exception as e:
                st.sidebar.error(f"Failed to create database tables: {str(e)}")
                return False
        
        # Always ensure vector extension and user_documents table exists
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        vector_setup = ensure_vector_extension_and_tables(SUPABASE_URL, SUPABASE_KEY)
        if not vector_setup:
            st.sidebar.warning("Vector database setup may not be complete")
            
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to initialize database: {str(e)}")
        return False

def ensure_vector_extension_and_tables(SUPABASE_URL, SUPABASE_KEY):
    """
    Ensure the pgvector extension is enabled and the user_documents table exists
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
        
    try:
        # Build the headers for direct REST API calls to Supabase
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        # Enable the pgvector extension
        enable_vector_sql = """
        CREATE EXTENSION IF NOT EXISTS vector;
        """
        
        # Create the user_documents table
        create_user_docs_sql = """
        CREATE TABLE IF NOT EXISTS user_documents (
            id UUID PRIMARY KEY,
            chat_id UUID NOT NULL,
            file_name TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(768),
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        );
        """
        
        # Create the match_documents function for vector similarity search
        create_match_function_sql = """
        CREATE OR REPLACE FUNCTION match_documents(
            query_embedding vector(768),
            match_chat_id uuid,
            match_threshold float DEFAULT 0.5,
            match_count int DEFAULT 5
        ) 
        RETURNS TABLE (
            id uuid,
            chat_id uuid,
            file_name text,
            chunk_index int,
            content text,
            similarity float
        ) 
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                ud.id,
                ud.chat_id,
                ud.file_name,
                ud.chunk_index,
                ud.content,
                1 - (ud.embedding <=> query_embedding) as similarity
            FROM
                user_documents ud
            WHERE
                ud.chat_id = match_chat_id
                AND 1 - (ud.embedding <=> query_embedding) > match_threshold
            ORDER BY
                ud.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        """
        
        # Execute SQL statements directly via REST API
        # First, enable the vector extension
        try:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/rpc/exec_sql",
                headers=headers,
                json={"query": enable_vector_sql}
            ).raise_for_status()
        except Exception as e:
            st.sidebar.warning(f"Failed to enable vector extension: {str(e)}")
        
        # Then create the user_documents table
        try:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/rpc/exec_sql",
                headers=headers,
                json={"query": create_user_docs_sql}
            ).raise_for_status()
        except Exception as e:
            st.sidebar.warning(f"Failed to create user_documents table: {str(e)}")
        
        # Finally, create the match function
        try:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/rpc/exec_sql",
                headers=headers,
                json={"query": create_match_function_sql}
            ).raise_for_status()
        except Exception as e:
            st.sidebar.warning(f"Failed to create match_documents function: {str(e)}")
        
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to set up vector database: {str(e)}")
        return False