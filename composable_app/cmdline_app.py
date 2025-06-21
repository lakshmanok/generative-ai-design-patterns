import logging

from composable_app.utils.guardrails import InputGuardrailException


## make sure logging starts first
def setup_logging(config_file: str = "logging.json"):
    import json
    import logging.config

    # Load the JSON configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Apply the configuration
    logging.config.dictConfig(config)
setup_logging()
##

import asyncio
from agents import task_assigner

async def async_input(user_prompt: str) -> str:
    result = await asyncio.to_thread(input, user_prompt)
    return result.strip()

async def app_main() -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting application")
    while True:
        topic = await async_input("Enter topic to write about (EMPTY to quit): ")
        if len(topic) == 0:
            return

        try:
            logger.info(f"Will ask AI agents to write about {topic}")
            assigner = task_assigner.TaskAssigner()
            article = await assigner.write_about(topic)
            print("******START*********\n")
            print(article.to_markdown())
            print("******END*********\n")
        except InputGuardrailException as e:
            logger.warning(e)
        except Exception as e:
            logger.error(e)

if __name__ == "__main__":
    asyncio.run(app_main())
