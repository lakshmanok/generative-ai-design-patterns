# install vllm with pip install vllm

# Standard library
import json
import time
import os
import torch

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vllm import LLM, SamplingParams

# Set environment variables for vLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only one GPU
os.environ["VLLM_USE_V1"] = "0"  # Force use of V0 engine for speculative decoding support
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduce memory fragmentation
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"  # Use fastest attention backend (if available)


# Use compatible Gemma-3 models that share the same vocabulary
BASE_MODEL = "google/gemma-3-27b-it"
SPECULATIVE_MODEL = "google/gemma-3-4b-it"

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def measure_throughput(llm, sampling_params, prompt, num_runs=10):
    """ Function to measure throughput of a model """

    total_tokens = 0
    total_time = 0
    tokens_per_second = []
    for _ in range(num_runs):
        start_time = time.time()

        # Run inference
        outputs = llm.generate([prompt], sampling_params)

        end_time = time.time()

        # Calculate tokens (input + output)
        input_tokens = len(outputs[0].prompt_token_ids)
        output_tokens = len(outputs[0].outputs[0].token_ids)
        total_tokens += input_tokens + output_tokens

        # Add inference time
        total_time += end_time - start_time
        tokens_per_second.append(total_tokens / (end_time - start_time))
    # Calculate metrics
    return tokens_per_second


if __name__ == '__main__':
    prompts = [
        "The future of AI is",
        "The black forest is located in",
        "The cathedral of Notre Dame is known for its",
    ]

    # Initialize baseline model (without speculative decoding) - use same base model for fair comparison
    baseline_llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.75,  # Reduced for 12B model to leave room for speculative model
        max_num_batched_tokens=16384,  # Increased for better GPU utilization
        max_num_seqs=128,  # Increased for higher throughput
        max_model_len=2048,  # Reasonable context length
        dtype="float16",  # Explicitly set dtype
        enforce_eager=False,  # Enable CUDA graphs for better performance
        enable_prefix_caching=True,  # Enable prefix caching for repeated prompts
    )

    # Set sampling parameters - optimized for speed
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,  # Add nucleus sampling for better quality/speed balance
        min_tokens=64,  # Reduced for faster testing
        max_tokens=256,  # Reduced for faster testing
        skip_special_tokens=True,  # Skip special tokens for faster decoding
    )

    # Run baseline measurements with fewer runs for faster testing
    print("Running baseline measurements...")
    baseline_results = []
    for prompt in prompts:
        print(f"Processing prompt: '{prompt[:30]}...'")
        baseline_results.extend(measure_throughput(baseline_llm, sampling_params, prompt, num_runs=3))

    # Clean up baseline model to free memory
    del baseline_llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")

    # Initialize speculative model with advanced optimizations
    print("Initializing speculative model with optimizations...")
    speculative_llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        speculative_config={
            "model": SPECULATIVE_MODEL,
            "num_speculative_tokens": 3,  # Increased since it's working well

        },
        trust_remote_code=True,
        gpu_memory_utilization=0.75,  # Reduced for 12B model to leave room for speculative model
        max_num_batched_tokens=16384,  # Increased for better GPU utilization
        max_num_seqs=128,  # Increased for higher throughput
        max_model_len=2048,  # Reasonable context length
        dtype="float16",  # Explicitly set dtype
        enforce_eager=False,  # Enable CUDA graphs for better performance
        enable_prefix_caching=True,  # Enable prefix caching for repeated prompts
    )

    # Run speculative measurements
    print("Running speculative measurements...")
    speculative_results = []
    for prompt in prompts:
        speculative_results.extend(measure_throughput(speculative_llm, sampling_params, prompt, num_runs=3))

    # Calculate statistics
    baseline_mean = np.mean(baseline_results)
    baseline_std = np.std(baseline_results)
    speculative_mean = np.mean(speculative_results)
    speculative_std = np.std(speculative_results)

    print(f"Baseline - Mean: {baseline_mean:.2f}, Std: {baseline_std:.2f}")
    print(f"Speculative - Mean: {speculative_mean:.2f}, Std: {speculative_std:.2f}")

    # Create distribution plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=baseline_results, label='Baseline', fill=True, alpha=0.3)
    sns.kdeplot(data=speculative_results, label='Speculative Decoding', fill=True, alpha=0.3)
    plt.xlabel('Tokens per Second')
    plt.ylabel('Density')
    plt.title('Distribution of Inference Speed: Baseline vs Speculative Decoding')
    plt.legend()
    plt.savefig('inference_speed_comparison.png')
    plt.close()

    # Save results to JSON
    try:
        with open("results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    # Update results
    data["baseline"] = baseline_results
    data["speculative_decoding"] = speculative_results

    # Save all result values to json
    with open("results.json", "w") as f:
        json.dump(data, f)
