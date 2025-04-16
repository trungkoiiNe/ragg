import os
import requests
import streamlit as st
from utils.chat_utils import search_documents

def get_llm_response(prompt, chat_id, supabase, temperature=0.7, max_tokens=1000):
    """
    Get a RAG-enhanced response from the model via OpenRouter API.
    
    Args:
        prompt (str): User's input prompt
        chat_id (str): Current chat session ID
        supabase: Supabase client for document search
        temperature (float): Temperature parameter for response generation
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: Model's response
    """
    # Get API key from environment variable
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # Set up headers following OpenRouter documentation
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://rag4all.app",  # Replace with your actual site URL in production
        "X-Title": "RAG4ALL Chat Application"
    }
    
    # Prepare messages for the API call
    messages = []
    
    # Search for relevant document context in the user's uploaded documents
    relevant_docs = []
    if chat_id:
        relevant_docs = search_documents(chat_id, prompt, supabase)
    
    # Format the system message with RAG context if available
    system_message = {
        "role": "system",
        "content": "You are a helpful AI assistant that provides informative responses."
    }
    
    # If we have relevant documents, include them in the system prompt
    if relevant_docs:
        context_str = "\n\n".join([doc["content"] for doc in relevant_docs])
        system_message["content"] = f"""You are a helpful AI assistant that provides informative responses.
        
You have access to the following context information from the user's documents:
---
{context_str}
---

Use this contextual information when relevant to provide accurate and helpful answers. 
If the user question is about information in the context, answer primarily based on the context.
If the context doesn't contain information to answer the question, say so and provide your best response based on your knowledge.
Don't explicitly mention that you're using context in your response unless asked about your sources.
"""
    
    # Add the system message
    messages.append(system_message)
    
    # Add conversation history if available
    if len(st.session_state.messages) > 0:
        # Convert previous messages to the format expected by the API
        # Only include the last few messages to stay within context window
        history = st.session_state.messages[-10:]  # Limit to last 10 messages
        messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in history])
        # Add the current prompt as the last user message if needed
        if history[-1]["role"] != "user" or history[-1]["content"] != prompt:
            messages.append({"role": "user", "content": prompt})
    else:
        # If no history, just add the current prompt
        messages.append({"role": "user", "content": prompt})
    
    # Prepare the request payload according to OpenRouter chat completions API
    data = {
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9
    }
    
    try:
        # Make the API request to the chat completions endpoint
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse and return the response content
        response_json = response.json()
        
        # Extract and return the generated message content
        return response_json["choices"][0]["message"]["content"]
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"Error: {str(e)}\n\nPlease make sure your API key is correctly set in the .env file."
        st.session_state.last_error = error_msg
        return error_msg