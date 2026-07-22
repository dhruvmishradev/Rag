from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Embedding Model
embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

# Load Chroma Database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# LLM
llm = ChatMistralAI(
    model="mistral-small-2506"
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use only the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG System Created")
print("Press 0 to exit")

while True:
    query = input("You: ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    print("\n===== Retrieved Documents =====\n")

    for i, doc in enumerate(docs, start=1):
        print(f"Document {i}")
        print(doc.page_content)
        print("-" * 60)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")