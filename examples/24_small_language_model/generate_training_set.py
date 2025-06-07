"""
Training Set Generation for Small Language Models

This script generates Python code examples using Claude and prepares them for training
a smaller language model through knowledge distillation. The training data is automatically
saved and can be reused for multiple training runs.

Key Features:
- Generates 1000 Python code examples using Claude
- Saves training data to JSON for reuse (training_data/python_code_examples.json)
- Loads existing training data if available
- Creates HuggingFace datasets ready for model training
- Supports knowledge distillation between teacher and student models

Usage:
1. Run the script to generate training data (if not already available)
2. Use the generated dataset for model training
3. For subsequent runs, the script will reuse saved training data

Example:
    python generate_training_set.py

    # Or in another script:
    from generate_training_set import load_dataset_from_saved_examples
    dataset = load_dataset_from_saved_examples()
"""

import os
import json
import time
from typing import List, Optional

import anthropic
from datasets import Dataset
from tqdm import tqdm

# Constants
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_TRAINING_DATA_PATH = "training_data/python_code_examples.json"
NUM_TRAINING_EXAMPLES = 10
RATE_LIMIT_DELAY = 1  # seconds between API calls
MAX_TOKEN_LENGTH = 1024
TRAIN_TEST_SPLIT_RATIO = 0.1

# ============================================================================
# Data Generation and Storage Functions
# ============================================================================

def save_training_examples(examples: List[str], filepath: str) -> None:
    """Save training examples to a JSON file for reuse."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    training_data = {
        "examples": examples,
        "metadata": {
            "num_examples": len(examples),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": CLAUDE_MODEL
        }
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Training examples saved to {filepath}")

def load_training_examples(filepath: str) -> Optional[List[str]]:
    """Load training examples from a JSON file."""
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = data.get("examples", [])
        metadata = data.get("metadata", {})

        print(f"Loaded {len(examples)} training examples from {filepath}")
        print(f"Generated at: {metadata.get('generated_at', 'Unknown')}")
        print(f"Model used: {metadata.get('model_used', 'Unknown')}")

        return examples
    except Exception as e:
        print(f"Error loading training examples: {str(e)}")
        return None

def generate_single_example(client: anthropic.Anthropic) -> Optional[str]:
    """Generate a single Python code example using Claude."""
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKEN_LENGTH,
            temperature=0.7,
            system="You are a helpful AI assistant that generates high-quality Python code examples.",
            messages=[{
                "role": "user",
                "content": "Generate a non-trivial but not too complex Python code example. "
                          "Focus on common programming patterns and best practices. "
                          "Return only the code, no documentation or explanation. "
                          "Make sure the code is complete and limited to 1000 tokens."
            }]
        )

        content = response.content[0].text.strip()

        # Remove markdown code block formatting
        if content.startswith("```python"):
            content = content[9:]
        if content.endswith("```"):
            content = content[:-3]

        return content.strip()

    except Exception as e:
        print(f"Error generating example: {str(e)}")
        return None

def generate_training_data(client: anthropic.Anthropic, num_examples: int) -> List[str]:
    """Generate Python code examples using Claude."""
    examples = []

    print(f"Generating {num_examples} training examples...")

    for i in tqdm(range(num_examples), desc="Generating examples", unit="example"):
        example = generate_single_example(client)
        if example:
            examples.append(example)

        # Rate limiting delay
        if i < num_examples - 1:  # No delay after the last example
            time.sleep(RATE_LIMIT_DELAY)

    print(f"Successfully generated {len(examples)} examples")
    return examples

# ============================================================================
# Dataset Creation Functions
# ============================================================================

def create_dataset(examples: List[str]) -> Dataset:
    """Convert the generated examples into a HuggingFace Dataset."""
    formatted_examples = []

    for code in examples:
        formatted_prompt = f"""Below is a Python code snippet. Please generate comprehensive documentation for it.

Code:
{code}

Documentation:"""

        formatted_examples.append({
            "text": formatted_prompt,
            "input": code
        })

    # Create and split the dataset
    dataset = Dataset.from_list(formatted_examples)
    dataset = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT_RATIO, seed=42)

    print(f"Created dataset with {len(dataset['train'])} training examples "
          f"and {len(dataset['test'])} validation examples")
    return dataset

def load_dataset_from_saved_examples(filepath: str = DEFAULT_TRAINING_DATA_PATH) -> Optional[Dataset]:
    """Load a HuggingFace Dataset from saved training examples."""
    examples = load_training_examples(filepath)
    if examples is None:
        return None
    return create_dataset(examples)

# ============================================================================
# Model Training Functions
# ============================================================================

def get_api_key() -> str:
    """Get Anthropic API key from environment or Colab userdata."""
    try:
        from google.colab import userdata
        return userdata.get('ANTHROPIC_API_KEY')
    except ImportError:
        from dotenv import load_dotenv
        load_dotenv("examples/keys.env")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return api_key


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to generate training data"""
    print("=" * 60)
    print("Training Set Generation for Small Language Models")
    print("=" * 60)

    # Check for existing training data
    print("\nChecking for existing training data...")
    examples = load_training_examples(DEFAULT_TRAINING_DATA_PATH)

    if examples is None or len(examples) < NUM_TRAINING_EXAMPLES:
        print("Generating new training data...")

        # Get API key and create client
        api_key = get_api_key()
        client = anthropic.Anthropic(api_key=api_key)

        # Generate training data
        examples = generate_training_data(client, NUM_TRAINING_EXAMPLES)
        save_training_examples(examples, DEFAULT_TRAINING_DATA_PATH)
    else:
        print("Using existing training data.")

    # Create dataset
    print("\nCreating dataset...")
    _ = create_dataset(examples)

    print(f"\n✅ Dataset ready with {len(examples)} examples")
    print("You can now proceed with model training using the generated dataset.")
    print(f"Training data saved at: {DEFAULT_TRAINING_DATA_PATH}")

if __name__ == "__main__":
    main()
