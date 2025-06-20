from .generic_writer_agent import GenericWriter
from .prompt_service import PromptService
from .article import Article
from .generic_writer_agent import Writer
from . import llms
from . import reviewer_panel
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
        writer = GenericWriter(await self.find_writer(topic))

        # Step 2: ask the writer to create an initial draft
        logger.info(f"Assigning {topic} to {writer.name()}")
        draft = await writer.write_about(topic)

        # Step 3: get the review panel to review the article
        logger.info("Sending article to review panel")
        panel_review = await reviewer_panel.get_panel_review_of_article(topic, draft)

        # Step 4: ask writer to rewrite article based on review
        article = await writer.revise_article(topic, draft, panel_review)
        return article

    async def find_writer(self, topic) -> Writer:
        prompt = PromptService.render_prompt("TaskAssigner_assign_writer",
                                             writers=[writer.name for writer in list(Writer)],
                                             topic=topic)
        result = await self.agent.run(prompt)
        return result.output


