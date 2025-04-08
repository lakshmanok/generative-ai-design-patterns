"""
Simple RAG System with Citations

This script implements a basic Retrieval-Augmented Generation (RAG) system
that retrieves relevant documents and uses them to answer questions with citations.
"""

import os
import json
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# Import our fixed embeddings class
from fixed_embedding import FixedHuggingFaceEmbeddings

# Constants
DOCUMENTS_DIR = "raw_texts/"
VECTOR_STORE_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"

def load_documents(directory_path: str) -> List[Any]:
    """Load documents from a directory."""
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents: List[Any]) -> List[Any]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks: List[Any]) -> Chroma:
    """Create a vector store from document chunks using the fixed embeddings."""
    embeddings = FixedHuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    vectorstore.persist()
    print(f"Created and persisted vector store to {VECTOR_STORE_DIR}")
    return vectorstore

def load_vector_store() -> Chroma:
    """Load an existing vector store using fixed embeddings."""
    embeddings = FixedHuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def needs_citation(content: str) -> bool:
    """Check if the content requires citations using OpenAI."""
    llm = ChatOpenAI(model_name=LLM_MODEL)
    prompt = PromptTemplate.from_template("""
    Check if the content requires citations. The return should be true or false in this JSON format: {{"requires_citations": true}}
    Content: {content}
    """)
    response = llm.invoke(prompt.format(content=content))
    return json.loads(response.content)["requires_citations"]

def check_sources(sentence: str) -> List[Any]:
    """Check if the sentence is in the document store using similarity_search."""
    vectorstore = load_vector_store()

    # Check if the vector store has any documents
    all_docs = vectorstore.get()
    if not all_docs or not all_docs.get('ids') or len(all_docs['ids']) == 0:
        print("WARNING: The vector store is empty. No documents found.")
        return []

    print(f"Vector store contains {len(all_docs['ids'])} documents.")

    try:
        chunks = vectorstore.similarity_search(sentence, k=5)
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        return []

    print(f"\nRetrieved {len(chunks)} chunks for sentence: '{sentence}'")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Content: {chunk.page_content[:100]}...")
        print(f"Metadata: {chunk.metadata}")

    return chunks

def check_for_citations(raw_response: str) -> List[Dict[str, Any]]:
    """Check if the response contains citations."""
    # Split the response into sentences
    sentences = raw_response.split(".")
    sentences = [sentence for sentence in sentences if sentence.strip()]

    review_sentences = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip() + "."
        print(f"Sentence {i}: {sentence}")

        review = needs_citation(sentence)
        chunks = check_sources(sentence) if review else []

        review_sentences.append({
            "sentence": sentence,
            "review": review,
            "chunks": chunks
        })

    return review_sentences

def create_rag_pipeline(vectorstore: Chroma) -> Any:
    """Create a simple RAG pipeline using similarity_search."""
    llm = ChatOpenAI(model_name=LLM_MODEL)

    def format_docs(docs: List[Any]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_docs(question: str) -> List[Any]:
        return vectorstore.similarity_search(question, k=4)

    template = """Answer the question based on the following context:

    CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS:
    1. Use information only from the provided context
    2. If the information isn't in the context, say "I don't have enough information to answer this question"
    3. Be concise and accurate in your response
    4. If information is available, write at least 5 sentences

    Your answer should be comprehensive and accurate.
    """

    rag_chain = (
        {"context": lambda q: format_docs(retrieve_docs(q)), "question": RunnablePassthrough()}
        | PromptTemplate.from_template(template)
        | llm
    )

    return rag_chain

def create_citation_mapping(review_sentences: List[Dict[str, Any]]) -> Dict[str, int]:
    """Create a mapping of file names to citation numbers."""
    file_to_citation = {}
    citation_counter = 1

    # Collect all unique file names and assign citation numbers
    for review_sentence in review_sentences:
        if review_sentence["review"] and len(review_sentence["chunks"]) > 0:
            file_references = set([x.metadata["source"] for x in review_sentence["chunks"]])
            for file_name in file_references:
                if file_name not in file_to_citation:
                    file_to_citation[file_name] = citation_counter
                    citation_counter += 1

    return file_to_citation

def format_citation(file_references: set, file_to_citation: Dict[str, int]) -> str:
    """Format a citation string for a set of file references."""
    citation = " ["
    citation_numbers = [str(file_to_citation[file_name]) for file_name in file_references]
    citation += ", ".join(citation_numbers) + "]"
    return citation

def add_references_section(response: str, file_to_citation: Dict[str, int]) -> str:
    """Add a references section to the response."""
    if file_to_citation:
        response += "\n\nReferences:\n"
        for file_name, citation_number in sorted(file_to_citation.items(), key=lambda x: x[1]):
            response += f"[{citation_number}] {file_name}\n"
    return response

def add_citations_to_response(review_sentences: List[Dict[str, Any]]) -> str:
    """Add citations to the response and create a references section."""
    # Create a mapping of file names to citation numbers
    file_to_citation = create_citation_mapping(review_sentences)

    # Write content with citations
    response_with_citations = ""
    for review_sentence in review_sentences:
        response_with_citations += review_sentence["sentence"]
        if review_sentence["review"] and len(review_sentence["chunks"]) == 0:
            response_with_citations += " [Citation needed] "
        elif review_sentence["review"] and len(review_sentence["chunks"]) > 0:
            # Get unique file references
            file_references = set([x.metadata["source"] for x in review_sentence["chunks"]])

            # Create citation with numbers
            citation = format_citation(file_references, file_to_citation)
            response_with_citations += citation
        response_with_citations += " "

    # Add references section at the end
    response_with_citations = add_references_section(response_with_citations, file_to_citation)

    return response_with_citations

def initialize_vector_store() -> Chroma:
    """Initialize the vector store, creating a new one if needed."""
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
        vectorstore = load_vector_store()

        # Check if the vector store has documents
        all_docs = vectorstore.get()
        if not all_docs or not all_docs.get('ids') or len(all_docs['ids']) == 0:
            print("WARNING: The loaded vector store is empty. Creating a new one...")
            if not os.path.exists(DOCUMENTS_DIR) or len(os.listdir(DOCUMENTS_DIR)) == 0:
                print(f"ERROR: {DOCUMENTS_DIR} directory doesn't exist or is empty")
                raise FileNotFoundError(f"{DOCUMENTS_DIR} directory doesn't exist or is empty")

            documents = load_documents(DOCUMENTS_DIR)
            chunks = split_documents(documents)
            return create_vector_store(chunks)
        else:
            print(f"Loaded vector store with {len(all_docs['ids'])} documents.")
            return vectorstore

def main():
    """Run the RAG system."""
    print("Loading RAG system...")

    # Make sure we have an API key set
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key

    try:
        # Initialize the vector store
        vectorstore = initialize_vector_store()

        # Create the RAG pipeline
        rag_chain = create_rag_pipeline(vectorstore)

        # Process a question
        question = "What are the Brandenburg Concertos?"
        print(f"\nProcessing question: {question}")
        response = rag_chain.invoke(question)
        print("\n" + response.content)

        # Add citations to the response
        review_sentences = check_for_citations(response.content)
        response_with_citations = add_citations_to_response(review_sentences)

        print("\nResponse with citations:")
        print(response_with_citations)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
