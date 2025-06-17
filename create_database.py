from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import chromadb
import os
import shutil

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data"
COLLECTION_NAME = "alice_collection"

def main():
    # Load documents directly with a specific encoding
    alice_file_path = os.path.join(DATA_PATH, "books", "alice_in_wonderland.md")
    
    if not os.path.exists(alice_file_path):
        print(f"File not found: {alice_file_path}")
        return
    
    print(f"Loading file: {alice_file_path}")
    
    try:
        # Try with latin-1 encoding which is more permissive
        with open(alice_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Create a proper Document object
        document = Document(page_content=text, metadata={"source": alice_file_path})
        print(f"Successfully loaded document with latin-1 encoding")
        
        # Split document - now passing a list of Document objects
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents([document])
        print(f"Split into {len(chunks)} chunks")
        
        # Clean existing database
        if os.path.exists(CHROMA_PATH):
            print(f"Removing existing database at {CHROMA_PATH}")
            shutil.rmtree(CHROMA_PATH)
        
        # Create database
        print("Creating embeddings and database...")
        embedding_function = OpenAIEmbeddings()
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME
        )
        
        # Explicitly persist
        db.persist()
        print(f"Database created and persisted to {CHROMA_PATH}")
        
        # Verify
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = client.list_collections()
        print(f"Collections after creation: {[c.name for c in collections]}")
        for coll in collections:
            print(f"Collection '{coll.name}' has {coll.count()} documents")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
