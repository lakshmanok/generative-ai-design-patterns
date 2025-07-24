## make sure logging starts first
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.google_genai import GoogleGenAI


def setup_logging(config_file: str = "logging.json"):
    import json
    import logging.config

    # Load the JSON configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Apply the configuration
    logging.config.dictConfig(config)
setup_logging()
##

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import os
from llama_index.core import StorageContext, Settings, load_index_from_storage
from composable_app.utils import llms

if __name__ == '__main__':
    Settings.embed_model = GoogleGenAIEmbedding(model_name=llms.EMBED_MODEL, api_key=os.environ["GEMINI_API_KEY"])
    Settings.llm = GoogleGenAI(model=llms.DEFAULT_MODEL, api_api_key=os.environ["GEMINI_API_KEY"])
    storage_context = StorageContext.from_defaults(persist_dir="data")
    index = load_index_from_storage(storage_context)

    query_engine = RetrieverQueryEngine.from_args(retriever=index.as_retriever(similarity_top_k=3))


    def semantic_rag(question):
        response = query_engine.query(question)
        response = {
            "answer": str(response),
            "source_nodes": response.source_nodes
        }
        print(response['answer'])
        for node in response['source_nodes']:
            print(node)
        return response

    semantic_rag("What is in-context learning?")
