import time
from typing import List
from vllm import LLM, SamplingParams

def generate_prompts(num_samples: int = 100) -> List[str]:
    """Generate a list of sample prompts."""
    return [
        "Write a short Python function that calculates the fibonacci sequence.",
        "Explain the concept of recursion in programming.",
        "What are the best practices for error handling in Python?",
        "How does garbage collection work in Python?",
        "Explain the difference between lists and tuples in Python."
    ] * (num_samples // 5)  # Repeat the 5 prompts to get desired number of samples

def process_individual(
    model: LLM,
    prompts: List[str],
    sampling_params: SamplingParams
) -> float:
    """Process prompts one by one and return total time."""
    start_time = time.time()

    for prompt in prompts:
        _ = model.generate(prompt, sampling_params)

    end_time = time.time()
    return end_time - start_time

def process_batched(
    model: LLM,
    prompts: List[str],
    sampling_params: SamplingParams
) -> float:
    """Process prompts in a single batch and return total time."""
    start_time = time.time()

    _ = model.generate(prompts, sampling_params)

    end_time = time.time()
    return end_time - start_time

def main():
    # Initialize the model
    model = LLM(
        model="google/gemma-3-1b-it",
        trust_remote_code=True,
        tensor_parallel_size=1,  # Changed from 4 to 1 to use a single GPU
        gpu_memory_utilization=0.9
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=100
    )

    # Generate sample prompts
    num_samples = 100
    prompts = generate_prompts(num_samples)

    # Warm up the model
    print("Warming up the model...")
    _ = model.generate(prompts[:1], sampling_params)

    # Process individually
    print("\nProcessing prompts individually...")
    individual_time = process_individual(model, prompts, sampling_params)
    individual_throughput = num_samples / individual_time

    # Process in batch
    print("\nProcessing prompts in batch...")
    batch_time = process_batched(model, prompts, sampling_params)
    batch_throughput = num_samples / batch_time

    # Print results
    print("\nResults:")
    print(f"Number of samples: {num_samples}")
    print(f"Individual processing time: {individual_time:.2f} seconds")
    print(f"Individual throughput: {individual_throughput:.2f} samples/second")
    print(f"Batch processing time: {batch_time:.2f} seconds")
    print(f"Batch throughput: {batch_throughput:.2f} samples/second")
    print(f"Speedup factor: {batch_throughput/individual_throughput:.2f}x")

if __name__ == "__main__":
    main()
