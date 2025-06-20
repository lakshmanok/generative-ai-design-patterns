from .generic_writer_agent import GenericWriter
from .prompt_service import PromptService
from .article import Article
from .generic_writer_agent import Writer
from . import llms
from dataclasses import dataclass
from pydantic_ai import Agent
import logging
import uuid

logger = logging.getLogger(__name__)

class TaskAssigner:
    def __init__(self):
        self.id = f"Task Assigner Agent {uuid.uuid4()}"
        system_prompt = PromptService.render_prompt("TaskAssigner_system_prompt")

        self.agent = Agent(llms.SMALL_MODEL,
                           output_type=Writer,
                           model_settings=llms.default_model_settings(),
                           retries=2,
                           system_prompt=system_prompt)

        logger.info(f"Created {self.id}")

    def name(self) -> str:
        return self.id

    async def write_about(self, topic: str) -> Article:
        # Step 1: Identify who can write on this topic
        prompt = PromptService.render_prompt("TaskAssigner_assign_writer",
                                             writers=list(Writer))
        result = await self.agent.run(prompt)
        writer = GenericWriter(result.output)

        # Step 2: ask the writer to create an initial draft
        logger.info(f"Assigning {topic} to {writer.name()}")
        article = await writer.write_about(topic)
        return article

