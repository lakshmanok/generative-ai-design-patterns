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


prompts = [
    "The future of AI is",
    "The black forest is located in",
    "The cathedral of Notre Dame is known for its",
]

# Initialize baseline model (without speculative decoding)
baseline_llm = LLM(
    model="google/gemma-3-4b-it",
    tensor_parallel_size=1,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,  # Reduced from 0.9
    max_num_batched_tokens=4096,  # Reduced from default
    max_num_seqs=256,  # Limit concurrent sequences
    dtype="float16",  # Explicitly set dtype
    enforce_eager=True,  # Disable CUDA graph
)

# Initialize speculative model
speculative_llm = LLM(
    model="google/gemma-3-4b-it",
    tensor_parallel_size=1,
    speculative_model="google/gemma-3-1b-it",
    num_speculative_tokens=5,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,  # Reduced from 0.9
    max_num_batched_tokens=4096,  # Reduced from default
    max_num_seqs=256,  # Limit concurrent sequences
    dtype="float16",  # Explicitly set dtype
    enforce_eager=True,  # Disable CUDA graph
)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    min_tokens=256,
    max_tokens=1024,
)

# Run baseline measurements
baseline_results = []
for prompt in prompts:
    baseline_results.extend(measure_throughput(baseline_llm, sampling_params, prompt))

# Run speculative measurements
speculative_results = []
for prompt in prompts:
    speculative_results.extend(measure_throughput(speculative_llm, sampling_params, prompt))

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
