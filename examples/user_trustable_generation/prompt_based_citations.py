"""
RAG with Prompt based Citations

This script implements a RAG system with prompt based citations.
It uses a language model to generate responses with in-line citations
based on retrieved documents from a vector store.
"""

import os
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

# Import our fixed embeddings class
from fixed_embedding import FixedHuggingFaceEmbeddings

# Constants
DOCUMENTS_DIR = "documents/"
VECTOR_STORE_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4

def load_documents(directory_path: str) -> List[Any]:
    """Load documents from a directory.

    Args:
        directory_path: Path to the directory containing documents

    Returns:
        List of loaded documents
    """
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents: List[Any]) -> List[Any]:
    """Split documents into smaller chunks.

    Args:
        documents: List of documents to split

    Returns:
        List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks: List[Any]) -> Chroma:
    """Create a vector store from document chunks using the fixed embeddings.

    Args:
        chunks: List of document chunks

    Returns:
        Chroma vector store
    """
    # Using fixed HuggingFace embeddings that can handle non-string inputs
    embeddings = FixedHuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    vectorstore.persist()
    print(f"Created and persisted vector store to {VECTOR_STORE_DIR}")
    return vectorstore

def load_vector_store() -> Chroma:
    """Load an existing vector store using fixed embeddings.

    Returns:
        Chroma vector store
    """
    embeddings = FixedHuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def format_docs(docs: List[Any]) -> str:
    """Format documents for context.

    Args:
        docs: List of documents

    Returns:
        Formatted document text
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_source_references(docs: List[Any]) -> List[Dict[str, Any]]:
    """Create a list of sources with identifiers.

    Args:
        docs: List of documents

    Returns:
        List of source dictionaries with IDs
    """
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

def format_sources_for_prompt(sources: List[Dict[str, Any]]) -> str:
    """Format sources for the prompt.

    Args:
        sources: List of source dictionaries

    Returns:
        Formatted sources text
    """
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

def create_scientific_rag_pipeline(vectorstore: Chroma) -> Any:
    """Create a RAG pipeline with in-line scientific-style citations.

    Args:
        vectorstore: Chroma vector store

    Returns:
        RAG chain
    """
    # Create a retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}  # Retrieve top K chunks
    )

    # Create a language model
    llm = ChatOpenAI(model_name=LLM_MODEL)

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

def ensure_api_key() -> None:
    """Ensure that the OpenAI API key is set in the environment."""
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
        print("API key set successfully.")

def initialize_vector_store() -> Chroma:
    """Initialize the vector store, creating a new one if needed.

    Returns:
        Chroma vector store
    """
    if not os.path.exists(VECTOR_STORE_DIR):
        if not os.path.exists(DOCUMENTS_DIR) or len(os.listdir(DOCUMENTS_DIR)) == 0:
            print(f"ERROR: {DOCUMENTS_DIR} directory doesn't exist or is empty")
            raise FileNotFoundError(f"{DOCUMENTS_DIR} directory doesn't exist or is empty")

        print("Creating new vector store from documents...")
        documents = load_documents(DOCUMENTS_DIR)
        chunks = split_documents(documents)
        return create_vector_store(chunks)
    else:
        print("Loading existing vector store...")
        return load_vector_store()

def process_question(rag_chain: Any, question: str) -> str:
    """Process a question using the RAG pipeline.

    Args:
        rag_chain: RAG chain
        question: Question to process

    Returns:
        Answer with citations
    """
    print("\nProcessing your question...")
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return f"Error: {str(e)}"

def main() -> None:
    """Run the RAG system."""
    print("Loading RAG system with prompt based citations...")

    ensure_api_key()

    # Initialize the vector store
    vectorstore = initialize_vector_store()

    # Create the RAG pipeline
    rag_chain = create_scientific_rag_pipeline(vectorstore)

    question = "What was Bach's music style?"
    answer = process_question(rag_chain, question)
    print("\n" + answer)

if __name__ == "__main__":
    main()
