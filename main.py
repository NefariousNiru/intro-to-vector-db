from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

import os

load_dotenv()

if __name__ == "__main__":
    print("Retrieving")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0)

    query = "What is Pinecone?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print("Result without RAG: ", result)

    vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
    rag_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})
    print("Result with RAG: ", result)