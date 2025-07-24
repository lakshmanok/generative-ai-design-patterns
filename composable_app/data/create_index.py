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
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from composable_app.utils import llms
import openparse
import logging
import os

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    Settings.embed_model = GoogleGenAIEmbedding(model_name=llms.EMBED_MODEL, api_key=os.environ["GEMINI_API_KEY"])
    Settings.llm = GoogleGenAI(model=llms.DEFAULT_MODEL, api_api_key=os.environ["GEMINI_API_KEY"])

    DOC_PATH = "data/book.pdf"
    if not os.path.exists(DOC_PATH):
        logger.error(f"Please place a file named {DOC_PATH} or use the checked-in index as-is")
        raise Exception(f"{DOC_PATH} not found")

    parser = openparse.DocumentParser()
    parsed_doc = parser.parse(DOC_PATH)
    nodes = parsed_doc.to_llama_index_nodes()
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir="data")

