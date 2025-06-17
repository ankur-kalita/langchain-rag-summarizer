import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import chromadb
import shutil

load_dotenv()

CHROMA_PATH = "chroma"
CSV_FILE_PATH = "data/files/zoomcar-opening-balance.csv"  
COLLECTION_NAME = "csv_collection" 

def main():
    # Check OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables or .env file")
        return
    else:
        print(f"OPENAI_API_KEY found: {api_key[:5]}...")
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f"CSV file not found: {CSV_FILE_PATH}")
        return
    
    # Load CSV file
    print(f"Loading CSV file: {CSV_FILE_PATH}")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        
        csv_text = df.to_string(index=False)
        
        rows_text = []
        for index, row in df.iterrows():
            row_text = f"Row {index}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
            rows_text.append(row_text)
        
        all_text = csv_text + "\n\n" + "\n".join(rows_text)
        
        # Create a Document object
        document = Document(
            page_content=all_text,
            metadata={"source": CSV_FILE_PATH}
        )
        
        print(f"Successfully loaded CSV data with {len(df)} rows")
        
        # Split document
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