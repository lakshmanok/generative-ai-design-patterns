"""
Helper functions for the LLM benchmark.
"""

import os

import argparse

from dotenv import load_dotenv


def load_api_key():
    """Load OpenAI API key from environment."""
    if os.path.exists("../saved_keys.env"):
        load_dotenv("../saved_keys.env")
    else:
        # Try loading from current directory
        if os.path.exists("saved_keys.env"):
            load_dotenv("saved_keys.env")
        else:
            print("Warning: saved_keys.env not found. Make sure OPENAI_API_KEY is set in environment.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return api_key

def parse_arguments():
    """Parse command-line arguments for benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="Run LLM inference performance benchmark against OpenAI API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms.",
        help="The prompt to send to the LLM"
    )

    parser.add_argument(
        "--num-users",
        type=int,
        default=5,
        help="Number of concurrent users to simulate"
    )

    parser.add_argument(
        "--requests-per-user",
        type=int,
        default=3,
        help="Number of requests each user will make"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for the benchmark"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum number of tokens to generate"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (0.0-2.0)"
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds"
    )

    parser.add_argument(
        "--ramp-up-time",
        type=float,
        default=5.0,
        help="Time to gradually add users in seconds"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.png",
        help="Output file path for the results plot"
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the results plot"
    )

    return parser.parse_args()
