"""
Example usage of the LLM benchmark tool.
"""

import asyncio
from typing import List

from llm_benchmark import LLMBenchmark, BenchmarkConfig, BenchmarkResults



# Example usage and utility functions

async def run_benchmark_example() -> BenchmarkResults:
    """
    Example demonstrating how to use the LLM benchmark tool.

    Returns:
        BenchmarkResults from the completed benchmark run
    """
    # Configuration for OpenAI API
    openai_config = BenchmarkConfig(
        endpoint_url="https://api.openai.com/v1/chat/completions",
        num_users=5,
        requests_per_user=3,
        prompt="Write a short story about a robot learning to paint.",
        model="gpt-3.5-turbo",
        max_tokens=100,
        api_key="your-openai-api-key-here"  # Replace with actual key
    )

    # Configuration for vLLM endpoint
    vllm_config = BenchmarkConfig(
        endpoint_url="http://localhost:8000/generate",
        num_users=10,
        requests_per_user=5,
        prompt="Explain the concept of machine learning in simple terms.",
        max_tokens=150,
        temperature=0.7
    )

    # Choose configuration (change to openai_config to test OpenAI)
    config = vllm_config

    async with LLMBenchmark(config) as benchmark:
        results = await benchmark.run_benchmark()
        benchmark.print_summary()
        benchmark.plot_results("benchmark_results.png")
        return results


async def compare_multiple_configs() -> List[BenchmarkResults]:
    """
    Compare performance across different benchmark configurations.

    Returns:
        List of BenchmarkResults for each configuration tested
    """
    configs = [
        BenchmarkConfig(
            endpoint_url="http://localhost:8000/generate",
            num_users=5,
            requests_per_user=3,
            prompt="Short prompt test",
            max_tokens=50
        ),
        BenchmarkConfig(
            endpoint_url="http://localhost:8000/generate",
            num_users=10,
            requests_per_user=3,
            prompt="Longer prompt test with more complex instructions and context",
            max_tokens=150
        )
    ]

    results = []
    for i, config in enumerate(configs):
        print(f"\nRunning configuration {i+1}...")
        async with LLMBenchmark(config) as benchmark:
            result = await benchmark.run_benchmark()
            benchmark.print_summary()
            results.append(result)

    return results


if __name__ == "__main__":
    # Run the example benchmark
    asyncio.run(run_benchmark_example())
