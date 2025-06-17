from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
import nltk
import chromadb

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
COLLECTION_NAME = "alice_collection"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No documents found. Check your data directory.")
        return
    
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    if not os.path.exists(DATA_PATH):
        print(f"Data directory '{DATA_PATH}' does not exist.")
        return []
    
    # Add encoding parameter to handle special characters
    loader = DirectoryLoader(DATA_PATH, glob="*.md", loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    if documents:
        print(f"First document preview: {documents[0].page_content[:200]}...")
    
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if len(chunks) > 10:
        document = chunks[10]
        print("Sample chunk:")
        print(document.page_content)
        print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    print(f"Creating new database at {os.path.abspath(CHROMA_PATH)}")
    
    try:
        # Use OpenAI embeddings
        embedding_function = OpenAIEmbeddings()
        
        db = Chroma.from_documents(
            documents=chunks,  
            embedding=embedding_function,  
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME
        )
        
        try:
            # Use chromadb client to verify
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collections = client.list_collections()
            
            if collections:
                collection = client.get_collection(COLLECTION_NAME)
                count = collection.count()
                print(f"Database created successfully with {count} documents")
            else:
                print("No collections found after creation")
                
        except Exception as verify_error:
            print(f"Could not verify database creation: {verify_error}")
            # Try alternative verification
            print("Database creation completed (verification failed)")
        
        # Check files created
        if os.path.exists(CHROMA_PATH):
            files = os.listdir(CHROMA_PATH)
            print(f"Database files created: {files}")
        else:
            print(f"WARNING: Database directory not found after creation!")
            
    except Exception as e:
        print(f"ERROR creating database: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
