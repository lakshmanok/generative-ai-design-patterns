import logging
from .prompt_service import PromptService
from . import llms
import uuid
from pydantic_ai import Agent

# Look at logging.json: all logs here go to logs/guards.json
logger = logging.getLogger(__name__)

class InputGuardrailException(Exception):
    def __init___(self, message):
        super().__init__(message)

class InputGuardrail:
    def __init__(self, name: str, condition: str, should_reject=True):
        self.id = f"Input Guardrail {name} {uuid.uuid4()}"
        self.system_prompt = PromptService.render_prompt("InputGuardrail_prompt",
                                                         condition=condition,
                                                         should_reject=should_reject)

        self.agent = Agent(llms.SMALL_MODEL,
                           output_type=bool,
                           model_settings=llms.default_model_settings(),
                           retries=2,
                           system_prompt=self.system_prompt)

        # logger.info(f"Created InputGuardrail {self.id}")

    async def is_acceptable(self, prompt: str, raise_exception=False) -> bool:
        result = await self.agent.run(prompt)

        logger.debug(f"Input checked by {self.id}", extra={
            "guardrail_id": self.id,
            "condition": self.system_prompt,
            "input": prompt,
            "output": result.output
        })

        if raise_exception and not result.output:
            raise InputGuardrailException(f"{self.id} failed on {prompt}")

        return result.output
