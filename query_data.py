import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
# Add this debugging code to query_data.py before accessing the specific collection
import chromadb
client = chromadb.PersistentClient(path="chroma")
collections = client.list_collections()
print(f"Available collections: {[c.name for c in collections]}")
for coll in collections:
    print(f"Collection '{coll.name}' has {coll.count()} documents")

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Use explicit model to avoid permission issues
    embedding_function = OpenAIEmbeddings()
    
    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        print(f"Database directory '{CHROMA_PATH}' does not exist. Please run create_database.py first.")
        return
    
    COLLECTION_NAME = "alice_collection"

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
            print("Database is empty. Please run create_database.py to populate it.")
            return
            
    except Exception as e:
        print(f"Error accessing database: {e}")
        return

    # Search with diagnostic info
    print(f"Searching for: '{query_text}'")
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    
    print(f"Found {len(results)} results")
    
    if len(results) == 0:
        print("No results found in database.")
        return
    
    # Show all results with scores
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1}: Score = {score:.3f}")
        print(f"Content preview: {doc.page_content[:150]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    
    # Use lower threshold for testing
    best_score = results[0][1]
    if best_score < 0.3:  # Lower threshold for testing
        print(f"\nBest relevance score ({best_score:.3f}) is below threshold.")
        print("Showing result anyway for debugging:")
        print(results[0][0].page_content)
        return

    # Continue with normal processing
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results[:3]])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response_text = model.invoke(prompt).content

    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results[:3]]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
