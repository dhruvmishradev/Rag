from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Documents
docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization algorithm that minimizes the loss function."),
    Document(page_content="Neural networks use Gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

# Embedding Model
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Create Chroma Vector Store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

# Similarity Retriever
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("\n===== Similarity Search Results =====\n")

similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)

# MMR Retriever
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

print("\n===== MMR Search Results =====\n")

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)