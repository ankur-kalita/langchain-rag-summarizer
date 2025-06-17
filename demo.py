from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import os
from dotenv import load_dotenv
load_dotenv()

CHROMA_PATH = "chroma_test"
docs = [Document(page_content="Hello world", metadata={"source": "test.md"})]
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH, collection_name="test_collection")
# db.persist()
print("Files in chroma_test:", os.listdir(CHROMA_PATH))
