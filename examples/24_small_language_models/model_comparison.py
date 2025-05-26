import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# Set CUDA launch blocking for better error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

@dataclass
class ModelResult:
    """Container for model generation results."""
    tokens_per_second: float
    response: str
    success: bool
    error_message: Optional[str] = None

def test_model(model_name: str, prompt: str) -> ModelResult:
    """Test a single model with the given prompt.

    Args:
        model_name: Name of the model to test
        prompt: Input prompt for generation

    Returns:
        ModelResult containing the test results
    """
    print(f"\nTesting {model_name}...")
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        # Prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate and measure time
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,  # Generate at least 1024 new tokens
            do_sample=True,  # Enable sampling for more diverse outputs
            temperature=0.7,  # Add some randomness but keep it focused
            top_p=0.9  # Nucleus sampling to maintain coherence
        )
        end_time = time.time()

        # Calculate metrics
        tokens_per_second = (len(outputs[0]) - len(inputs.input_ids[0])) / (end_time - start_time)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        return ModelResult(
            tokens_per_second=tokens_per_second,
            response=response,
            success=True
        )

    except Exception as e:
        print(f"Error with model: {str(e)}")
        return ModelResult(
            tokens_per_second=0.0,
            response="",
            success=False,
            error_message=str(e)
        )

def run_model_comparison(model_names: Tuple[str, str], prompt: str) -> None:
    """Run a comparison between two models."""
    # Test both models
    result1 = test_model(model_names[0], prompt)
    result2 = test_model(model_names[1], prompt)

    # Print results
    for model_name, result in zip(model_names, [result1, result2]):
        if result.success:
            print(f"\nModel: {model_name}")
            print(f"Generated {result.tokens_per_second:.2f} tokens/second")
            print("\nGenerated Response:")
            print("=" * 80)
            # Remove the prompt from the response if it's included
            response_text = result.response
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
            # Print the response with proper wrapping
            if not response_text.strip():
                print("  No response generated")
            else:
                for line in response_text.split('\n'):
                    if line.strip():
                        print(f"  {line.strip()}")
            print("=" * 80)
        else:
            print(f"\nModel: {model_name}")
            print(f"Error: {result.error_message}")

    # Print comparison
    print("\nPerformance Comparison:")
    print(f"{model_names[0]}: {result1.tokens_per_second:.2f} tokens/second")
    print(f"{model_names[1]}: {result2.tokens_per_second:.2f} tokens/second")
    if result2.tokens_per_second > 0:
        print(f"Speed ratio: {result1.tokens_per_second/result2.tokens_per_second:.2f}x")

def main():
    """Main function to run the model comparison."""
    # Model names
    model_names = ("google/gemma-3-1b-it", "google/gemma-3-4b-it")

    # Complex prompt that might challenge smaller models
    complex_prompt = """Given the following complex scenario:
    A quantum computer with 1000 qubits is trying to factor a 2048-bit RSA number while simultaneously solving a protein folding problem.
    The quantum computer is also running a machine learning model to optimize its own quantum gates.
    Explain the potential quantum interference patterns and how they might affect the protein folding simulation.
    Consider both the quantum decoherence effects and the classical-quantum interface limitations."""

    run_model_comparison(model_names, complex_prompt)

if __name__ == "__main__":
    main()

