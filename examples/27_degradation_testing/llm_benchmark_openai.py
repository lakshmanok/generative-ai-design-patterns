"""
Run a benchmark against the OpenAI API.
"""

import asyncio
from llm_benchmark_helpers import parse_arguments, load_api_key
from llm_benchmark import LLMBenchmark, BenchmarkConfig

async def main():
    """Main function to run the benchmark with parsed arguments."""
    args = parse_arguments()

    try:
        api_key = load_api_key()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading API key: {e}")
        return

    # Configure benchmark with command-line arguments
    config = BenchmarkConfig(
        endpoint_url="https://api.openai.com/v1/chat/completions",
        api_key=api_key,
        model=args.model,
        prompt=args.prompt,
        num_users=args.num_users,
        requests_per_user=args.requests_per_user,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        ramp_up_time=args.ramp_up_time
    )

    print("Starting OpenAI API benchmark with configuration:")
    print(f"  Model: {config.model}")
    print(f"  Users: {config.num_users}")
    print(f"  Requests per user: {config.requests_per_user}")
    print(f"  Max tokens: {config.max_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Prompt: {config.prompt[:50]}{'...' if len(config.prompt) > 50 else ''}")
    print()

    # Run benchmark
    async with LLMBenchmark(config) as benchmark:
        _ = await benchmark.run_benchmark()
        benchmark.print_summary()

        if not args.no_plot:
            benchmark.plot_results(args.output)
            print(f"Results plot saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
