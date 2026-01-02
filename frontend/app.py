import streamlit as st
import requests
import base64
import os

# --- Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.write("Upload PDFs to train the AI.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Upload & Process"):
            with st.spinner("Uploading and processing... (this may take a moment)"):
                try:
                    # Send file to Backend API
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Success! Added {data.get('chunks_added')} text chunks and {data.get('images_added')} images.")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to Backend. Is it running?")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    st.divider()
    
    # Check Backend Health
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if health.status_code == 200:
            status = health.json()
            if status.get("vector_store_loaded"):
                st.success("🟢 System Online & Ready")
            else:
                st.warning("🟡 System Online (Empty Database)")
        else:
            st.error("🔴 Backend Error")
    except:
        st.error("🔴 Backend Offline")

# --- Main Chat Interface ---
st.title("🤖 Chat with Documents (Multimodal)")
st.markdown("Ask questions about your PDFs. I can read text and analyze diagrams/images.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("What is the total cost in the invoice?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"query": prompt}
                response = requests.post(f"{BACKEND_URL}/query", json=payload)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer received.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                elif response.status_code == 503:
                    st.warning("Please upload a document first.")
                else:
                    st.error(f"Server Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend server. Please ensure `python backend/app.py` is running.")