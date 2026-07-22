from dotenv import load_dotenv
import os

from huggingface_hub import login
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Login to Hugging Face using the token from .env
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"}
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"}
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_Book"}
    ),
]

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vectorstore.similarity_search("what is used for data analysis ? " ,k = 2)


for r in result :
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)