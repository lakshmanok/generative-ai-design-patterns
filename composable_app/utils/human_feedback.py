import logging

logger = logging.getLogger(__name__)

def record_human_feedback(target, ai_input, ai_response, human_choice):
    logger.info(f"HumanFeedback", extra={
        "target": target,
        "ai_input": ai_input,
        "ai": ai_response,
        "human": human_choice,
    })
