import os
from dotenv import load_dotenv
import logging

BEST_MODEL="gemini-2.0-flash"
DEFAULT_MODEL="gemini-2.0-flash"
SMALL_MODEL="gemini-2.5-flash-lite-preview-06-17"

logger = logging.getLogger(__name__)

def _setup():
    source_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(f"{source_dir}/../keys.env")
    assert os.environ["GEMINI_API_KEY"][:2] == "AI", \
        "Please specify the GEMINI_API_KEY access token in keys.env file or as an environment variable."

    logger.info(f"Defaulting to {DEFAULT_MODEL}; will use {BEST_MODEL} "
                f"for higher quality and {SMALL_MODEL} for lower latency")

def default_model_settings():
    from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings
    model_settings = GeminiModelSettings(
                temperature=0.25,
                gemini_safety_settings=[
                    {
                        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                        'threshold': 'BLOCK_ONLY_HIGH',
                    }
                ]
            )
    return model_settings

# run on module load
_setup()
