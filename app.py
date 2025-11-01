import streamlit as st
import os
from pinecone import Pinecone
from openai import OpenAI
import hashlib
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chat", layout="wide", initial_sidebar_state="expanded")

# Helper functions
def load_projects():
    if os.path.exists("projects.json"):
        with open("projects.json", "r") as f:
            return json.load(f)
    return {}

# Initialize clients
@st.cache_resource
def init_clients():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)
    return pc, openai_client, index

pc, openai_client, index = init_clients()

# Session state initialization
if "projects" not in st.session_state:
    st.session_state.projects = load_projects()
if "current_project" not in st.session_state:
    st.session_state.current_project = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}


def save_projects():
    with open("projects.json", "w") as f:
        json.dump(st.session_state.projects, f, indent=2)

def save_chat_history(project_name):
    history_file = f"chat_history_{project_name}.json"
    with open(history_file, "w") as f:
        json.dump(st.session_state.chat_history.get(project_name, []), f, indent=2)

def load_chat_history(project_name):
    history_file = f"chat_history_{project_name}.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    return []

def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
        dimensions=512
    )
    return response.data[0].embedding

def chunk_text(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Sidebar navigation
with st.sidebar:
    st.title("📚 RAG Projects")
    page = st.radio("Navigation", ["Create Project", "Chat"], label_visibility="collapsed")
    
    st.divider()
    
    if st.session_state.projects:
        st.subheader("Your Projects")
        for proj_name in st.session_state.projects.keys():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(proj_name, key=f"proj_{proj_name}", use_container_width=True):
                    st.session_state.current_project = proj_name
                    if proj_name not in st.session_state.chat_history:
                        st.session_state.chat_history[proj_name] = load_chat_history(proj_name)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{proj_name}"):
                    del st.session_state.projects[proj_name]
                    if os.path.exists(f"chat_history_{proj_name}.json"):
                        os.remove(f"chat_history_{proj_name}.json")
                    save_projects()
                    st.rerun()

# Page: Create Project
if page == "Create Project":
    st.title("📄 Create New RAG Project")
    
    project_name = st.text_input("Project Name", placeholder="Enter a unique name for this project")
    
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Maximum total size: 100MB"
    )
    
    chunk_size = st.slider("Chunk Size (characters)", 500, 2000, 1000, 100)
    
    if st.button("Process & Create Project", type="primary"):
        if not project_name:
            st.error("Please provide a project name")
        elif project_name in st.session_state.projects:
            st.error("Project name already exists")
        elif not uploaded_files:
            st.error("Please upload at least one document")
        else:
            # Check total size
            total_size = sum([file.size for file in uploaded_files])
            if total_size > 100 * 1024 * 1024:
                st.error("Total file size exceeds 100MB")
            else:
                with st.spinner("Processing documents..."):
                    from document_processor import process_documents
                    
                    try:
                        # Process documents
                        all_chunks = process_documents(uploaded_files, chunk_size)
                        
                        # Create embeddings and upsert to Pinecone
                        progress_bar = st.progress(0)
                        vectors = []
                        
                        for idx, chunk in enumerate(all_chunks):
                            embedding = get_embedding(chunk["text"])
                            vector_id = f"{project_name}_{hashlib.md5(chunk['text'].encode()).hexdigest()}"
                            
                            vectors.append({
                                "id": vector_id,
                                "values": embedding,
                                "metadata": {
                                    "project": project_name,
                                    "text": chunk["text"],
                                    "source": chunk["source"],
                                    "chunk_index": chunk["chunk_index"]
                                }
                            })
                            
                            # Batch upsert every 100 vectors
                            if len(vectors) >= 100:
                                index.upsert(vectors=vectors)
                                vectors = []
                            
                            progress_bar.progress((idx + 1) / len(all_chunks))
                        
                        # Upsert remaining vectors
                        if vectors:
                            index.upsert(vectors=vectors)
                        
                        # Save project
                        st.session_state.projects[project_name] = {
                            "created_at": datetime.now().isoformat(),
                            "num_chunks": len(all_chunks),
                            "files": [f.name for f in uploaded_files]
                        }
                        save_projects()
                        
                        st.success(f"✅ Project '{project_name}' created successfully with {len(all_chunks)} chunks!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")

# Page: Chat
elif page == "Chat":
    if not st.session_state.current_project:
        st.info("👈 Select a project from the sidebar to start chatting")
    else:
        project_name = st.session_state.current_project
        st.title(f"💬 Chat: {project_name}")
        
        # Load chat history for current project
        if project_name not in st.session_state.chat_history:
            st.session_state.chat_history[project_name] = load_chat_history(project_name)
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history[project_name]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Clear chat button
        if st.session_state.chat_history[project_name]:
            if st.button("Clear Chat History"):
                st.session_state.chat_history[project_name] = []
                save_chat_history(project_name)
                st.rerun()
        
        # Chat input
        user_query = st.chat_input("Ask a question about your documents...")
        
        if user_query:
            # Add user message
            st.session_state.chat_history[project_name].append({
                "role": "user",
                "content": user_query
            })
            
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get query embedding
                    query_embedding = get_embedding(user_query)
                    
                    # Search Pinecone
                    results = index.query(
                        vector=query_embedding,
                        top_k=5,
                        include_metadata=True,
                        filter={"project": {"$eq": project_name}}
                    )
                    
                    # Build context from results
                    context_chunks = [match["metadata"]["text"] for match in results["matches"]]
                    context = "\n\n".join(context_chunks)
                    
                    # Build conversation history for context
                    conversation_context = ""
                    recent_history = st.session_state.chat_history[project_name][-6:]  # Last 3 exchanges
                    for msg in recent_history[:-1]:  # Exclude current query
                        conversation_context += f"{msg['role'].title()}: {msg['content']}\n"
                    
                    # Generate response with GPT-4
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Use the conversation history to maintain context, but primarily answer based on the retrieved documents."},
                        {"role": "user", "content": f"Previous conversation:\n{conversation_context}\n\nContext from documents:\n{context}\n\nCurrent question: {user_query}"}
                    ]
                    
                    response = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        temperature=0.7
                    )
                    
                    assistant_response = response.choices[0].message.content
                    st.write(assistant_response)
                    
                    # Add assistant response to history
                    st.session_state.chat_history[project_name].append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    
                    # Save chat history
                    save_chat_history(project_name)