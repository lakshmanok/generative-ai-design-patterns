"""
Prompt management using Jinja2. See: https://github.com/pydantic/pydantic-ai/issues/921#issuecomment-2813030935
"""
from jinja2 import Environment, FileSystemLoader
from typing import Any
import logging

# reads Jinja2 files from the TEMPLATE_DIR
TEMPLATE_DIR = "prompts"

# Look at logging.json: all logs here go to logs/prompts.json
logger = logging.getLogger(__name__)

# render template
class PromptService:
    @staticmethod
    def render_prompt(prompt_name, **variables: Any) -> str:
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template(f"{prompt_name}.j2")
        prompt = template.render(**variables)

        extra_args = dict(**variables)
        extra_args['prompt_name'] = prompt_name
        logger.info(prompt, extra=extra_args)
        return prompt

