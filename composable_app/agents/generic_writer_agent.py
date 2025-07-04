"""
Agent that can write content on any topic.
"""
import uuid
from typing import Any

from pydantic_ai import Agent
import logging

from pydantic_ai.agent import AgentRunResult

from composable_app.utils import llms
from .article import Article
from composable_app.utils.prompt_service import PromptService
from enum import Enum, auto
from composable_app.utils import long_term_memory as ltm
from composable_app.utils import save_for_eval as evals


logger = logging.getLogger(__name__)

## the types of writers
class AutoName(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name

# HISTORIAN intentionally omitted for illustration
# although hardcoded here, this might be in user-settings
content_type = {
    "GENERALIST": "short article",
    "MATH_WRITER": "detailed solution"
}

class Writer(AutoName):
    HISTORIAN = auto()
    MATH_WRITER = auto()
    GENERALIST = auto()

def get_content_type(writer: Writer):
    # because HISTORIAN is not in the content_type dict, it will default to "2 paragraphs"
    return content_type.get(writer.name, "2 paragraphs")

class GenericWriter:
    def __init__(self, writer: Writer):
        self.id = f"{writer} Agent {uuid.uuid4()}"
        self.writer = writer
        # different prompts for different Writer
        system_prompt_file = f"{self.writer.name}_system_prompt".lower()
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
        # the prompt is the same for all writers, but the content_type is parameterized in this file
        prompt_vars = {
            "prompt_name": f"GenericWriter_write_about",
            "content_type": get_content_type(self.writer),
            "additional_instructions": ltm.search_relevant_memories(f"{self.writer.name}, write about {topic}"),
            "topic": topic
        }
        prompt = PromptService.render_prompt(**prompt_vars)
        result = await self.agent.run(prompt)
        logger.info(result.usage())
        await evals.record_ai_response("initial_draft",
                                       ai_input=prompt_vars,
                                       ai_response=result.output)
        return result.output

    async def revise_article(self, topic: str, initial_draft: Article, panel_review: str) -> Article:
        # the prompt is the same for all writers
        prompt_vars = {
            "prompt_name": "GenericWriter_revise_article",
            "topic": topic,
            "content_type": get_content_type(self.writer),
            "additional_instructions": ltm.search_relevant_memories(f"{self.writer.name}, revise {topic}"),
            "initial_draft": initial_draft.to_markdown(),
            "panel_review": panel_review
        }
        prompt = PromptService.render_prompt(**prompt_vars)
        result: AgentRunResult[Article] = await self.agent.run(prompt)
        logger.info(result.usage())
        await evals.record_ai_response("revised_draft",
                                       ai_input=prompt_vars,
                                       ai_response=result.output)
        return result.output
