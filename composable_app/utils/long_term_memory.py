import json
import tempfile
import os
from typing import List, Dict, Any
import logging
from composable_app.utils import llms
from mem0 import Memory

logger = logging.getLogger(__name__)

class LongTermMemory:
    def __init__(self, app_name: str = "composable_app"):
        # In reality, you'd reuse the same directory so that you maintain state across runs
        temp_dir = tempfile.mkdtemp(prefix=app_name)
        logger.warning(f"In reality, you should not use temporary directory for mem0 storage: {temp_dir}")

        vectordb = os.path.join(temp_dir, "vectordb")
        logger.info(f"Saving embeddings into {vectordb}")

        config = {
            "vector_store_old": {
                "provider": "chroma",
                "config": {
                    "collection_name": app_name,
                    "path": vectordb,
                }
            },
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": llms.DEFAULT_MODEL,
                    "temperature": 0.2,
                    "max_tokens": 1000,
                    "top_p": 1.0
                }
            },
            "embedder": {
                "provider": "gemini",
                "config": {
                    "model": "models/gemini-embedding-exp-03-07",
                    "embedding_dims": 1536
                }
            },
            "history_db_path": os.path.join(temp_dir, "history.db"),
            "version": "v1.1"
        }
        try:
            logger.debug("Initializing mem0 ...")
            self.memory = Memory.from_config(config)
            logger.debug("✓ Mem0 initialized")
        except Exception as e:
            logger.error(f"Error initializing memory: {e}")
            raise e

    def add_to_memory(self,
                      user_message: str,
                      metadata: Dict,
                      user_id: str = "default_user") -> Dict[str, Any]:
        try:
            messages = [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            result = self.memory.add(messages=messages, user_id=user_id, metadata=metadata)
            logger.debug(f"✓ Memory add for {user_id} successful: {result}")
            return result
        except Exception as e:
            logger.error(f"Error adding to memory", e)
            return {"error": str(e)}

    def search_relevant_memories(self,
                                 query: str,
                                 user_id: str = "default_user", limit: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant memories based on query"""
        logger.debug(f"Searching memories for: '{query}'")
        try:
            memories = self.memory.search(query=query, user_id=user_id, limit=limit)

            # Handle different return formats from mem0
            processed_memories = []

            if isinstance(memories, list):
                for i, mem in enumerate(memories):
                    if isinstance(mem, dict):
                        # Standard dictionary format
                        if 'memory' in mem:
                            processed_memories.append(mem)
                        elif 'text' in mem:
                            # Alternative format with 'text' key
                            processed_memories.append({'memory': mem['text'], **mem})
                        else:
                            # Unknown dict format, try to find content
                            content = str(mem)
                            processed_memories.append({'memory': content})
                    elif isinstance(mem, str):
                        # String format
                        processed_memories.append({'memory': mem})
                    else:
                        # Other formats, convert to string
                        content = str(mem)
                        processed_memories.append({'memory': content})

            elif isinstance(memories, dict):
                # Check if it's a response with results
                if 'results' in memories:
                    return _process_search_results(memories['results'], query)
                else:
                    # Single memory dict
                    processed_memories = [memories]
            else:
                logger.error(f"Unexpected memories format: {type(memories)}")
                return []

            logger.debug(f"Found {len(processed_memories)} relevant memories")
            return processed_memories

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

def _process_search_results(results: List, query: str) -> List[Dict[str, Any]]:
    """Helper function to process search results from a results array"""
    processed_memories = []
    for i, mem in enumerate(results):
        if isinstance(mem, dict):
            processed_memories.append(mem)
        else:
            processed_memories.append({'memory': str(mem)})
    return processed_memories

# for use in the app
mem0 = LongTermMemory()

def add_to_memory(user_message: str, metadata: Dict, user_id: str = "default_user") -> Dict[str, Any]:
    return mem0.add_to_memory(user_message, metadata, user_id)

def search_relevant_memories(query: str, user_id: str = "default_user", limit: int = 3) -> List[str]:
    memories = mem0.search_relevant_memories(query, user_id, limit)
    return [m['memory'] for m in memories]
