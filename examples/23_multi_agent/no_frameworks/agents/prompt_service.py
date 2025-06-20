"""
Prompt management using Jinja2. See: https://github.com/pydantic/pydantic-ai/issues/921#issuecomment-2813030935
"""
from cookiecutter.prompt import prompt_for_config
from jinja2 import Environment, FileSystemLoader
from typing import Any
import logging

TEMPLATE_DIR = "prompts"
logger = logging.getLogger(__name__)

# render template
class PromptService:
    @staticmethod
    def render_prompt(prompt_name, **variables: Any) -> str:
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template(f"{prompt_name}.j2")
        prompt = template.render(**variables)
        logger.debug(prompt)
        return prompt

