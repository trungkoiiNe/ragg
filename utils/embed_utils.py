import streamlit as st
import httpx
import requests

def get_embeddings_from_ollama(texts, model="nomic-embed-text"):
    """
    Get embeddings for a list of texts using Ollama's nomic-embed-text model
    
    Args:
        texts (list): List of text strings to embed
        model (str): Embedding model to use
        
    Returns:
        list: List of embeddings (vectors)
    """
    embeddings = []
    
    # Using httpx for better async support
    with httpx.Client(timeout=60.0) as client:
        for text in texts:
            try:
                response = client.post(
                    "http://localhost:11434/api/embeddings",
                    json={
                        "model": model,
                        "prompt": text
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "embedding" in result:
                        embeddings.append(result["embedding"])
                    else:
                        st.warning(f"No embedding found in the response: {result}")
                        embeddings.append([0.0] * 768)  # Default empty embedding
                else:
                    st.warning(f"Failed to get embedding: {response.text}")
                    embeddings.append([0.0] * 768)  # Default empty embedding
                    
            except Exception as e:
                st.warning(f"Error getting embeddings: {str(e)}")
                embeddings.append([0.0] * 768)  # Default empty embedding
                
    return embeddings

def check_ollama_server():
    """
    Check if the Ollama server is running
    
    Returns:
        bool: True if server is running, False otherwise
    """
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except Exception:
        return False