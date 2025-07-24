## make sure logging starts first
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
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from composable_app.utils import llms

if __name__ == '__main__':
    Settings.embed_model = GoogleGenAIEmbedding(model_name=llms.EMBED_MODEL, api_key=os.environ["GEMINI_API_KEY"])
    Settings.llm = GoogleGenAI(model=llms.DEFAULT_MODEL, api_api_key=os.environ["GEMINI_API_KEY"])

    documents = SimpleDirectoryReader(input_dir="data",
                                      required_exts=[".pdf"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="data")

