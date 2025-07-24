"""
Agent that can write content on any topic.
"""
import uuid
from abc import ABC, abstractmethod
from dataclasses import replace
from pydantic_ai import Agent
import logging
import os

from composable_app.utils import llms
from .article import Article
from composable_app.utils.prompt_service import PromptService
from enum import Enum, auto
from composable_app.utils import long_term_memory as ltm
from composable_app.utils import save_for_eval as evals

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.google_genai import GoogleGenAI

logger = logging.getLogger(__name__)

## the types of writers
class AutoName(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name

class Writer(AutoName):
    HISTORIAN = auto()
    MATH_WRITER = auto()
    GENAI_WRITER = auto()
    GENERALIST = auto()

class AbstractWriter(ABC):
    def __init__(self, writer: Writer):
        self.id = f"{writer} Agent {uuid.uuid4()}"
        self.writer = writer
        logger.info(f"Created {self.id}")

    def name(self) -> str:
        return self.id

    @abstractmethod
    async def write_response(self, topic: str, prompt: str) -> Article:
        pass

    @abstractmethod
    async def revise_response(self, prompt: str) -> Article:
        pass

    @abstractmethod
    def get_content_type(self) -> str:
        pass

    async def write_about(self, topic: str) -> Article:
        # the prompt is the same for all writers, but the content_type is parameterized in this file
        prompt_vars = {
            "prompt_name": f"AbstractWriter_write_about",
            "content_type": self.get_content_type(),
            "additional_instructions": ltm.search_relevant_memories(f"{self.writer.name}, write about {topic}"),
            "topic": topic
        }
        prompt = PromptService.render_prompt(**prompt_vars)
        result = await self.write_response(topic, prompt)
        await evals.record_ai_response("initial_draft",
                                       ai_input=prompt_vars,
                                       ai_response=result)
        return result

    async def revise_article(self, topic: str, initial_draft: Article, panel_review: str) -> Article:
        # the prompt is the same for all writers
        prompt_vars = {
            "prompt_name": "AbstractWriter_revise_article",
            "topic": topic,
            "content_type": self.get_content_type(),
            "additional_instructions": ltm.search_relevant_memories(f"{self.writer.name}, revise {topic}"),
            "initial_draft": initial_draft.to_markdown(),
            "panel_review": panel_review
        }
        prompt = PromptService.render_prompt(**prompt_vars)
        result: Article = await self.revise_response(prompt)
        await evals.record_ai_response("revised_draft",
                                       ai_input=prompt_vars,
                                       ai_response=result)
        return result

class ZeroshotWriter(AbstractWriter):
    def __init__(self, writer: Writer):
        super().__init__(writer)
        # the prompt files are named by the writer name
        system_prompt_file = f"{self.writer.name}_system_prompt".lower()
        system_prompt = PromptService.render_prompt(system_prompt_file)

        self.agent = Agent(llms.BEST_MODEL,
                           output_type=Article,
                           model_settings=llms.default_model_settings(),
                           retries=2,
                           system_prompt=system_prompt)

    async def write_response(self, topic: str, prompt: str) -> Article:
        result = await self.agent.run(prompt)
        logger.info(result.usage())
        return result.output

    async def revise_response(self, prompt: str) -> Article:
        result = await self.agent.run(prompt)
        logger.info(result.usage())
        return result.output

    @abstractmethod
    def get_content_type(self) -> str:
        pass

class MathWriter(ZeroshotWriter):
    def __init__(self):
        super().__init__(Writer.MATH_WRITER)

    def get_content_type(self) -> str:
        return "detailed solution"

class HistoryWriter(ZeroshotWriter):
    def __init__(self):
        super().__init__(Writer.HISTORIAN)

    def get_content_type(self) -> str:
        return "2 paragraphs"

class GeneralistWriter(ZeroshotWriter):
    def __init__(self):
        super().__init__(Writer.GENERALIST)

    def get_content_type(self) -> str:
        return "short article"

"""
Use RAG on the contents of the book
"""
class GenAIWriter(ZeroshotWriter):
    def __init__(self):
        super().__init__(Writer.GENAI_WRITER)
        Settings.embed_model = GoogleGenAIEmbedding(model_name=llms.EMBED_MODEL, api_key=os.environ["GEMINI_API_KEY"])
        source_dir = os.path.dirname(os.path.abspath(__file__))
        storage_context = StorageContext.from_defaults(persist_dir=f"{source_dir}/../data")
        index = load_index_from_storage(storage_context)
        self.retriever = index.as_retriever(similarity_top_k=3)

    async def write_response(self, topic: str, prompt: str) -> Article:
        # semantic RAG
        nodes = self.retriever.retrieve(topic)
        prompt += f"\n**INFORMATION YOU CAN USE**\n{nodes}"
        result = await self.agent.run(prompt)
        logger.info(result.usage())
        article = result.output

        # add page reference
        pages = [str(node.metadata['bbox'][0]['page']) for node in nodes]
        article = replace(article, full_text=article.full_text + f"\nSee pages: {', '.join(pages)}")
        return article

    def get_content_type(self):
        return "2 paragraphs"

class WriterFactory:
    @staticmethod
    def create_writer(writer: Writer) -> AbstractWriter:
        match writer:
            case Writer.MATH_WRITER.name:
                return MathWriter()
            case Writer.HISTORIAN.name:
                return HistoryWriter()
            case Writer.GENAI_WRITER:
                return GenAIWriter()
            case _:
                return GeneralistWriter()
