import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

CHROMA_PATH = "chroma"
COLLECTION_NAME = "csv_collection"  # Match the collection name from create_csv_database.py

PROMPT_TEMPLATE = """
Answer the question based only on the following financial data:

{context}

---

Answer the question based on the above financial data: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Debug collections
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = client.list_collections()
    print(f"Available collections: {[c.name for c in collections]}")
    for coll in collections:
        print(f"Collection '{coll.name}' has {coll.count()} documents")

    # Use explicit model to avoid permission issues
    embedding_function = OpenAIEmbeddings()
    
    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        print(f"Database directory '{CHROMA_PATH}' does not exist. Please run create_csv_database.py first.")
        return
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    
    # Check database contents
    try:
        collection_count = db._collection.count()
        print(f"Database contains {collection_count} documents")
        
        if collection_count == 0:
            print("Database is empty. Please run create_csv_database.py to populate it.")
            return
            
    except Exception as e:
        print(f"Error accessing database: {e}")
        return

    # Search with diagnostic info
    print(f"Searching for: '{query_text}'")
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0:
        print("No results found in database.")
        return
    
    # Continue with normal processing
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(model="gpt-4o-mini")
    response_text = model.invoke(prompt).content

    print(f"Response: {response_text}")

if __name__ == "__main__":
    main()