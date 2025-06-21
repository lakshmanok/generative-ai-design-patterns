"""
Agent that can write content on any topic.
"""
import uuid

from pydantic_ai import Agent
import logging

from composable_app.utils import llms
from .article import Article
from composable_app.utils.prompt_service import PromptService
from enum import Enum, auto

logger = logging.getLogger(__name__)

## the types of writers
class AutoName(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name

class Writer(AutoName):
    HISTORIAN = auto()
    MATH_WRITER = auto()
    GENERALIST = auto()


class GenericWriter:
    def __init__(self, writer: Writer):
        self.id = f"{writer} Agent {uuid.uuid4()}"
        system_prompt_file = f"{writer.value}_system_prompt".lower()
        system_prompt = PromptService.render_prompt(system_prompt_file)

        self.agent = Agent(llms.BEST_MODEL,
                           output_type=Article,
                           model_settings=llms.default_model_settings(),
                           retries=2,
                           system_prompt=system_prompt)

        logger.info(f"Created {self.id}")

    def name(self) -> str:
        return self.id

    async def write_about(self, topic: str) -> Article:
        # the prompt is the same for all writers
        prompt = PromptService.render_prompt("GenericWriter_write_about",
                                             topic=topic)
        result = await self.agent.run(prompt)
        logger.info(result.usage())
        return result.output

    async def revise_article(self, topic: str, initial_draft: Article, panel_review: str) -> Article:
        # the prompt is the same for all writers
        prompt = PromptService.render_prompt("GenericWriter_revise_article",
                                             topic=topic,
                                             initial_draft=initial_draft.to_markdown(),
                                             panel_review=panel_review)
        result = await self.agent.run(prompt)
        logger.info(result.usage())
        return result.output
