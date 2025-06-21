import os
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from pydantic import SecretStr

load_dotenv()

if __name__ == "__main__":
    print("Ingesting with FAISS")

    loader = PyPDFLoader("faiss.pdf")
    document = loader.load()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0, api_key=SecretStr(os.environ["OPENAI_API_KEY"]), model=os.environ["MODEL"])
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_local")       # Persist

    # allow dangerous deserialization is true because we pickled the vectors into .pkl file
    vectorstore = FAISS.load_local("faiss_local", embeddings, allow_dangerous_deserialization=True)

    rag_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combined_docs = create_stuff_documents_chain(llm, rag_prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combined_docs)

    query = "What are the search types the paper tells about when searching using ANNs?"
    res = retrieval_chain.invoke(input={"input": query})

    print(res["answer"])




