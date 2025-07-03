import logging

logger = logging.getLogger(__name__)

def record_ai_response(target, ai_input, ai_response):
    logger.info(f"AI Response", extra={
        "target": target,
        "ai_input": ai_input,
        "ai_response": ai_response,
    })
