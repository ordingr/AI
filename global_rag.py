import os
import streamlit as st
import pandas as pd
from docx import Document
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import base64
import PyPDF2

# Debug mode flag
DEBUG_MODE = False


# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Set up OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

import PyPDF2

def read_file(file):
    """Read contents of various file types."""
    if DEBUG_MODE:
        print(f"DEBUG: Reading file: {file.name}")
    
    file_extension = file.name.split('.')[-1].lower()
    content = ""
    if file_extension == 'txt':
        content = file.getvalue().decode('utf-8')
    elif file_extension == 'csv':
        df = pd.read_csv(file)
        content = df.to_string()
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file)
        content = df.to_string()
    elif file_extension in ['doc', 'docx']:
        doc = Document(file)
        content = ' '.join([para.text for para in doc.paragraphs])
    elif file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfReader(file)
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"
    else:
        content = "Unsupported file format"
    
    if DEBUG_MODE:
        print(f"DEBUG: File content (first 200 chars): {content[:200]}")
    
    return content

def create_embedding(text):
    """Create embedding using OpenAI."""
    if DEBUG_MODE:
        print(f"DEBUG: Creating embedding for text: {text[:100]}...")
    
    embedding = openai_ef([text])[0]
    
    if DEBUG_MODE:
        print(f"DEBUG: Embedding created, shape: {len(embedding)}")
    
    return embedding

def add_to_chroma(text, metadata=None):
    """Add document to ChromaDB."""
    if DEBUG_MODE:
        print(f"DEBUG: Adding to ChromaDB. Text length: {len(text)}, Metadata: {metadata}")
    
    embedding = create_embedding(text)
    collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata] if metadata else None,
        ids=[str(collection.count() + 1)]
    )
    
    if DEBUG_MODE:
        print(f"DEBUG: Document added to ChromaDB. New collection count: {collection.count()}")

def similarity_search(query, k=5, ids=None):
    """Perform similarity search on ChromaDB using cosine similarity."""
    print(f"DEBUG: Performing similarity search. Query: {query}, k: {k}, ids: {ids}")
    
    query_embedding = create_embedding(query)
    
    try:
        if ids:
            # If specific IDs are provided, fetch those documents first
            print(f"DEBUG: Fetching documents with IDs: {ids}")
            specific_docs = collection.get(ids=ids, include=['documents', 'metadatas', 'embeddings'])
            print(f"DEBUG: Fetched documents: {specific_docs}")
            
            if not specific_docs['ids']:
                print("DEBUG: No documents found with the specified IDs")
                return [], [], []
            
            # Perform similarity search on the fetched documents
            similarities = [cosine_similarity([query_embedding], [doc_embedding])[0][0] 
                            for doc_embedding in specific_docs['embeddings']]
            
            # Sort results by similarity (descending order)
            sorted_results = sorted(zip(specific_docs['documents'], specific_docs['metadatas'], similarities), 
                                    key=lambda x: x[2], reverse=True)
        else:
            # If no specific IDs, perform a regular similarity search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            similarities = [1 - dist for dist in distances]
            sorted_results = list(zip(documents, metadatas, similarities))
        
        print(f"DEBUG: Search results: {len(sorted_results)} documents found")
        for i, (doc, meta, sim) in enumerate(sorted_results, 1):
            print(f"DEBUG: Result {i}: Similarity: {sim:.4f}, Metadata: {meta}, Document preview: {doc[:100]}...")
        
        return [r[0] for r in sorted_results], [r[1] for r in sorted_results], [r[2] for r in sorted_results]
    except Exception as e:
        print(f"DEBUG: Error in similarity search: {str(e)}")
        return [], [], []

def get_ai_response(query, context, model):
    """Get AI-generated response using OpenAI's chat completion."""
    if DEBUG_MODE:
        print(f"DEBUG: Getting AI response. Query: {query}, Context length: {len(context)}, Model: {model}")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question accurately and concisely."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        
        ai_response = response.choices[0].message.content
        
        if DEBUG_MODE:
            print(f"DEBUG: AI response generated: {ai_response}")
        
        return ai_response
    except Exception as e:
        if DEBUG_MODE:
            print(f"DEBUG: Error in OpenAI API call: {str(e)}")
        return f"Error in OpenAI API call: {str(e)}"

def print_chroma_contents():
    """Print the contents of the ChromaDB collection."""
    print("DEBUG: Checking ChromaDB contents")
    try:
        all_docs = collection.get()
        print(f"Total documents in ChromaDB: {len(all_docs['ids'])}")
        for id, doc, meta in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas']):
            print(f"Document ID: {id}, Metadata: {meta}")
            print(f"Document preview: {doc[:100]}...")
            print("---")
    except Exception as e:
        print(f"Error checking ChromaDB contents: {str(e)}")

def get_embedding_options():
    all_docs = collection.get()
    return [f"{meta['filename']} (ID: {id})" for id, meta in zip(all_docs['ids'], all_docs['metadatas'])]

def set_background():
    # Set page background
    set_png_as_page_bg('images/RAG_SQL2.webp')
    
    # Additional styling
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-color: #f0f0f0;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    .stApp {
        background-color: rgba(255, 255, 255, 0.1);
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }

    /* Revert text input to dark theme */
    .stTextInput>div>div>input {
        background-color: rgba(28, 31, 46, 0.7);
        color: white;
        border-color: rgba(128, 128, 128, 0.4);
    }

    /* Improve text readability */
    .stMarkdown, .stText {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }

    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }

    /* Style for the query input label */
    .stTextInput label {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }
    </style>
    """, unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), url("data:image/png;base64,{bin_str}");
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-color: #f0f0f0;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    set_background()  # Set the custom background

    st.title("Database RAG Application")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Retrieval-Augmented Generation with OpenAI and ChromaDB</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Settings")

    # Model selection with "gpt-4o-mini" as default
    model_option = st.sidebar.selectbox(
        "Choose OpenAI Model",
        ("gpt-4o-mini", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct", "gpt-4", "gpt-4-0125-preview"),
        index=0  # This sets the first option (gpt-4o-mini) as default
    )

    # File upload section
    st.sidebar.subheader("Upload a File")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['txt', 'csv', 'xls', 'xlsx', 'doc', 'docx', 'pdf'])
    
    if uploaded_file is not None:
        file_contents = read_file(uploaded_file)
        st.sidebar.write(f"File: {uploaded_file.name}")
        if st.sidebar.button("Add to Database"):
            add_to_chroma(file_contents, {"filename": uploaded_file.name})
            st.sidebar.success("File added to the database!")
            # Refresh the embedding options
            doc_options = get_embedding_options()
    
    # Embedding selection
    st.sidebar.subheader("Select Embeddings")
    doc_options = get_embedding_options()
    selected_docs = st.sidebar.multiselect("Select embeddings to include in search:", doc_options)
    selected_ids = [doc.split("(ID: ")[-1][:-1] for doc in selected_docs]

    # Delete selected embeddings
    if st.sidebar.button("Delete Selected Embeddings"):
        if selected_ids:
            try:
                collection.delete(ids=selected_ids)
                st.sidebar.success(f"Successfully deleted {len(selected_ids)} embedding(s).")
                # Refresh the embedding options
                doc_options = get_embedding_options()
                selected_docs = []
                selected_ids = []
            except Exception as e:
                st.sidebar.error(f"Error deleting embeddings: {str(e)}")
        else:
            st.sidebar.warning("No embeddings selected for deletion.")

    # Main area
    # Query the database
    st.subheader("Query the Database")
    query = st.text_input("Enter your query:")
    if query:
        results, metadatas, similarities = similarity_search(query, k=5, ids=selected_ids if selected_ids else None)
        
        if results:
            # We're not displaying the top similar documents anymore
            context = "\n\n".join(results)
            
            # Get AI-generated response
            try:
                ai_response = get_ai_response(query, context, model_option)
                st.subheader("AI Response:")
                st.write(ai_response)
            except Exception as e:
                st.error(f"Error generating AI response: {str(e)}")
        else:
            st.warning("No matching documents found. Try adjusting your query or adding more documents to the database.")

    # Debug information (optional)
    if DEBUG_MODE:
        st.subheader("Debug Information")
        st.write(f"Number of documents in the collection: {collection.count()}")
        st.write(f"Selected document IDs: {selected_ids}")
        if query:
            st.write(f"Query: {query}")
            st.write(f"Number of results: {len(results) if 'results' in locals() else 0}")

if __name__ == "__main__":
    main()

