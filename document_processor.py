import PyPDF2
import docx
from io import BytesIO

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = docx.Document(BytesIO(file.read()))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file):
    """Extract text from TXT file"""
    return file.read().decode('utf-8')

def process_documents(uploaded_files, chunk_size=1000):
    """Process multiple documents and return chunks"""
    all_chunks = []
    
    for file in uploaded_files:
        file_type = file.name.split('.')[-1].lower()
        
        # Extract text based on file type
        if file_type == 'pdf':
            text = extract_text_from_pdf(file)
        elif file_type == 'docx':
            text = extract_text_from_docx(file)
        elif file_type == 'txt':
            text = extract_text_from_txt(file)
        else:
            continue
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size)
        
        # Add metadata to chunks
        for idx, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                all_chunks.append({
                    "text": chunk,
                    "source": file.name,
                    "chunk_index": idx
                })
    
    return all_chunks

def chunk_text(text, chunk_size=1000):
    """Split text into chunks of specified size"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks