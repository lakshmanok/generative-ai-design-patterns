"""
Agents review content.
"""
import uuid

from pydantic_ai import Agent
import logging

from composable_app.utils import llms
from .article import Article
from composable_app.utils.prompt_service import PromptService
from composable_app.utils import save_for_eval as evals
from enum import Enum, auto
from typing import List, Tuple

logger = logging.getLogger(__name__)

class Reviewer(Enum):
    DISTRICT_REP = auto() # 1
    GRAMMAR_REVIEWER = auto()
    CONSERVATIVE_PARENT = auto()
    LIBERAL_PARENT = auto()
    SCHOOL_ADMIN = auto()

class ReviewerAgent:
    def __init__(self, reviewer: Reviewer):
        self.reviewer = reviewer
        self.id = f"{reviewer} Agent {uuid.uuid4()}"
        system_prompt_file = f"{reviewer.name}_system_prompt".lower()
        system_prompt = PromptService.render_prompt(system_prompt_file)

        self.agent = Agent(llms.DEFAULT_MODEL,
                           output_type=str,
                           model_settings=llms.default_model_settings(),
                           retries=2,
                           system_prompt=system_prompt)

        logger.info(f"Created {self.id}")

    def name(self) -> str:
        return self.id

    def reviewer_type(self) -> Reviewer:
        return self.reviewer

    async def review(self, topic: str, article: Article, reviews_so_far: List[Tuple[Reviewer, str]]) -> str:
        reviews_text = []
        for reviewer, review in reviews_so_far:
            reviews_text.append(f"BEGIN review by {reviewer.name}:\n{review}\nEND review\n")

        prompt_vars = {
            "prompt_name": "ReviewerAgent_review_prompt",
            "topic": topic,
            "article": article,
            "reviews": reviews_text
        }
        prompt = PromptService.render_prompt(**prompt_vars)
        result = await self.agent.run(prompt)
        logger.info(result.usage())
        await evals.record_ai_response(f"{self.reviewer.name}_review",
                                       ai_input=prompt_vars,
                                       ai_response=result.output)
        return result.output

class PanelSecretary:
    def __init__(self):
        self.id = f"PanelSecretary {uuid.uuid4()}"
        system_prompt = PromptService.render_prompt("secretary_system_prompt")

        self.agent = Agent(llms.DEFAULT_MODEL,
                           output_type=str,
                           model_settings=llms.default_model_settings(),
                           retries=2,
                           system_prompt=system_prompt)

        logger.info(f"Created {self.id}")

    def name(self) -> str:
        return self.id

    async def consolidate(self, topic: str, article: Article, reviews_so_far: List[Tuple[Reviewer, str]]) -> str:
        reviews_text = []
        for reviewer, review in reviews_so_far:
            reviews_text.append(f"BEGIN review by {reviewer.name}:\n{review}\nEND review\n")

        prompt_vars = {
            "prompt_name": "Secretary_consolidate_reviews",
            "topic": topic,
            "article": article,
            "reviews": reviews_text
        }
        prompt = PromptService.render_prompt(**prompt_vars)
        result = await self.agent.run(prompt)
        logger.info(result.usage())
        await evals.record_ai_response("consolidated_review",
                                       ai_input=prompt_vars,
                                       ai_response=result.output)
        return result.output

async def get_panel_review_of_article(topic: str, article: Article) -> str:
    first_round_reviews = await do_first_round_reviews(article, topic)
    final_reviews = await do_second_round_reviews(article, first_round_reviews, topic)
    return await summarize_reviews(article, final_reviews, topic)


async def summarize_reviews(article, final_reviews, topic) -> str:
    # Step 3: finally, the secretary summarizes
    secretary = PanelSecretary()
    summary_review = await secretary.consolidate(topic, article, final_reviews)
    return summary_review


async def do_second_round_reviews(article, first_round_reviews, topic) -> list:
    # Step 2: then they get to review again knowing what the others think
    review_panel = [ReviewerAgent(reviewer) for reviewer in list(Reviewer)[:-1]]
    final_reviews = list()
    for reviewer_agent in review_panel:
        review = await reviewer_agent.review(topic, article, first_round_reviews)
        final_reviews.append((reviewer_agent.reviewer_type(), review))
    return final_reviews


async def do_first_round_reviews(article, topic) -> list:
    # Step 1: each person on the panel reviews separately
    # set up a panel of reviewers: this is everyone except secretary
    review_panel = [ReviewerAgent(reviewer) for reviewer in list(Reviewer)[:-1]]
    first_round_reviews = list()
    for reviewer_agent in review_panel:
        review = await reviewer_agent.review(topic, article, reviews_so_far=[])
        first_round_reviews.append((reviewer_agent.reviewer, review))
    return first_round_reviews