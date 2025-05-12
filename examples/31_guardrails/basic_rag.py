from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.anthropic import Anthropic
import gutenberg_text_loader as gtl
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Indexer:
    """
    A class to load documents into LlamaIndex using BM25.
    
    Attributes:
        chunk_size (int): Size of text chunks for processing.
        chunk_overlap (int): Overlap between text chunks.
        docstore (SimpleDocumentStore): Document store for storing processed documents.
    """
    
    def __init__(
        self,
        cache_dir: str = "./.cache",
        chunk_size: int = 1024,
        chunk_overlap: int = 20
    ):
        """
        Initialize the Indexer.
        
        Args:
            chunk_size (int): Size of text chunks for processing. Defaults to 1024.
            chunk_overlap (int): Overlap between text chunks. Defaults to 20.
        """        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize a simple document store
        self.docstore = SimpleDocumentStore()
        
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        logger.info("Indexer initialized")
    

    def add_document_to_index(self, document: Document):
        # Parse the document into nodes
        nodes = self.node_parser.get_nodes_from_documents([document])

        # Add nodes to the document store
        self.docstore.add_documents(nodes)

        logger.info(f"Successfully loaded text from {document.id_} -- {len(nodes)} nodes created.")
            
    def get_docstore(self) -> SimpleDocumentStore:
        return self.docstore

def build_query_engine(model_id: str, urls: [str], chunk_size: int) -> RetrieverQueryEngine:
    gs = gtl.GutenbergSource()
    index = Indexer(chunk_size=chunk_size, chunk_overlap=chunk_size//10)
    
    for url in urls:
        doc = gs.load_from_url(url)
        index.add_document_to_index(doc)
    
    retriever = BM25Retriever.from_defaults(
        docstore=index.get_docstore(),
        similarity_top_k=5)
    
    llm = Anthropic(
        model=model_id,
        api_key=os.environ['ANTHROPIC_API_KEY'],
        temperature=0.2
    )
    
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever, llm=llm
    )
    
    return query_engine

def print_response_to_query(query_engine: RetrieverQueryEngine, query: str):
    response = query_engine.query(query)
    response = {
        "answer": str(response),
        "source_nodes": response.source_nodes
    }
    print(response['answer'])
    print("\n\n**Sources**:")
    for node in response['source_nodes']:
        print(node)
        