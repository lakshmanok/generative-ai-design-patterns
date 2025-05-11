import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

if os.path.exists("examples/keys.env"):
    load_dotenv("examples/keys.env")
else:
    raise FileNotFoundError("examples/keys.env not found")


def load_model(model_name, quantization_config=None):
    """Load a model with optional quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=100):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    generation_time = time.time() - start_time
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, generation_time

def main():
    model_name = "google/gemma3-1b-it"

    # Load original model
    print("Loading original model...")
    original_model, tokenizer = load_model(model_name)

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load quantized model
    print("Loading quantized model...")
    quantized_model, _ = load_model(model_name, quantization_config)

    # Test prompt
    prompt = "Explain quantum computing in simple terms."

    # Generate responses
    print("\nGenerating response from original model...")
    original_response, original_time = generate_response(original_model, tokenizer, prompt)

    print("\nGenerating response from quantized model...")
    quantized_response, quantized_time = generate_response(quantized_model, tokenizer, prompt)

    # Print results
    print("\n=== Original Model Response ===")
    print(original_response)
    print(f"Generation time: {original_time:.2f} seconds")

    print("\n=== Quantized Model Response ===")
    print(quantized_response)
    print(f"Generation time: {quantized_time:.2f} seconds")

    # Print memory usage
    print("\n=== Memory Usage ===")
    print(f"Original model memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Quantized model memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
