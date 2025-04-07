"""
Scientific RAG with In-Line Citations

This script implements a RAG system with scientific-style in-line citations
directly next to facts rather than in a separate section.
"""

import os
import json
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

# Import our fixed embeddings class
from fixed_embedding import FixedHuggingFaceEmbeddings

def load_documents(directory_path):
    """Load documents from a directory."""
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Create a vector store from document chunks using the fixed embeddings."""
    # Using fixed HuggingFace embeddings that can handle non-string inputs
    embeddings = FixedHuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create and persist the vector store
    persist_directory = "chroma_db"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"Created and persisted vector store to {persist_directory}")
    return vectorstore

def load_vector_store():
    """Load an existing vector store using fixed embeddings."""
    embeddings = FixedHuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_directory = "chroma_db"
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

def create_scientific_rag_pipeline(vectorstore):
    """Create a RAG pipeline with in-line scientific-style citations."""
    # Create a retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 chunks
    )

    # Create a language model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    # Function to format documents for context
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Function to create a list of sources with identifiers
    def create_source_references(docs):
        sources = []
        for i, doc in enumerate(docs):
            # Create a source identifier
            source_id = i + 1
            source = {
                "id": source_id,
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source)
        return sources

    # Create a scientific citation prompt template
    template = """Answer the question based on the following sources, using in-line citations like in scientific papers.

    SOURCES:
    {sources}

    QUESTION: {question}

    INSTRUCTIONS:
    1. Use information only from the provided sources
    2. Provide an answer with in-line citations using brackets, e.g., "Einstein developed the theory of relativity [1]"
    3. Use the source ID in brackets [1], [2], etc., corresponding to the source number
    4. Cite ALL facts with their source IDs
    5. If multiple sources support a fact, you can include multiple citations [1][2]
    6. If the information isn't in the sources, say "I don't have enough information to answer this question"
    7. Include a "References" section at the end listing all the sources you cited

    Your answer should be comprehensive, accurate, and include citations for all factual claims.
    """

    # Generate a prompt that includes properly formatted sources
    def format_sources_for_prompt(sources):
        formatted_sources = ""
        for source in sources:
            source_id = source["id"]
            source_info = f"[{source_id}] "

            # Add metadata if available
            if "metadata" in source and "source" in source["metadata"]:
                source_info += f"From: {source['metadata']['source']}\n"
            else:
                source_info += "Source document\n"

            # Add content preview
            content_preview = source["content"][:200] + "..." if len(source["content"]) > 200 else source["content"]
            source_info += f"Content: {content_preview}\n\n"

            formatted_sources += source_info

        return formatted_sources

    # Create a retriever chain that returns document references
    retriever_chain = retriever | create_source_references

    # Create the RAG chain using RunnableParallel for the inputs
    input_processor = RunnableParallel(
        sources=retriever_chain,
        question=RunnablePassthrough()
    )

    # Create the full chain
    rag_chain = (
        input_processor
        | (lambda x: {"sources": format_sources_for_prompt(x["sources"]), "question": x["question"]})
        | PromptTemplate.from_template(template)
        | llm
        | StrOutputParser()
    )

    return rag_chain

def run_interactive_session():
    """Run an interactive session with the scientific RAG system."""
    print("Loading scientific RAG system with in-line citations...")

    # Make sure we have an API key set
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key

    try:
        # Check if we should create a new vector store or load an existing one
        if not os.path.exists("chroma_db"):
            directory_path = "documents/"
            if not os.path.exists(directory_path) or len(os.listdir(directory_path)) == 0:
                print(f"ERROR: {directory_path} directory doesn't exist or is empty")
                return

            print("Creating new vector store from documents...")
            documents = load_documents(directory_path)
            chunks = split_documents(documents)
            vectorstore = create_vector_store(chunks)
        else:
            print("Loading existing vector store...")
            vectorstore = load_vector_store()

        # Create the RAG pipeline
        rag_chain = create_scientific_rag_pipeline(vectorstore)

        print("\nScientific RAG System loaded successfully!")
        print("Type 'exit' to quit.\n")

        # Interactive loop
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == 'exit':
                print("Goodbye!")
                break

            # Process the question
            print("\nProcessing your question...")
            try:
                answer = rag_chain.invoke(question)
                print("\n" + answer)
            except Exception as e:
                print(f"Error processing question: {str(e)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Run the scientific RAG system."""
    run_interactive_session()

if __name__ == "__main__":
    main()
