# RAG Chat Application with Pinecone

A Streamlit-based RAG (Retrieval Augmented Generation) application that allows you to upload documents, create projects, and chat with your data using OpenAI and Pinecone.

## Features

- 📄 Upload multiple documents (PDF, TXT, DOCX) up to 100MB total
- 🗂️ Create multiple RAG projects
- 💬 Chat with your documents using conversational AI
- 💾 Persistent chat history across sessions
- 🎯 Context-aware responses using vector search

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**
   - Create `.env` file
   - Add your API keys:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `PINECONE_API_KEY`: Your Pinecone API key
     - `PINECONE_INDEX_NAME`: Your Pinecone index name

3. **Create Pinecone Index:**
   - Log in to Pinecone console
   - Create a new index with:
     - Dimension: 512 (for text-embedding-3-small)
     - Metric: cosine

4. **Run the application:**
```bash
streamlit run app.py
```

## Usage

### Creating a Project

1. Navigate to "Create Project" page
2. Enter a unique project name
3. Upload your documents (PDF, TXT, or DOCX)
4. Adjust chunk size if needed (default: 1000 characters)
5. Click "Process & Create Project"

### Chatting with Documents

1. Select a project from the sidebar
2. The chat interface will open
3. Ask questions about your documents
4. The AI maintains conversation history for context-aware responses

### Managing Projects

- Click project name in sidebar to open chat
- Click 🗑️ to delete a project
- Use "Clear Chat History" to reset conversations

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── document_processor.py   # Document processing utilities
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (to create manually)
├── projects.json           # Project metadata (auto-generated)
└── chat_history_*.json     # Chat histories (auto-generated)
```

## Notes

- Chat histories are saved locally and persist across sessions
- Each project's data is stored in Pinecone with project-specific metadata
- The app uses GPT-4 for responses and text-embedding-3-small for embeddings
