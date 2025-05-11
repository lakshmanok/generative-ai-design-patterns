import os
import json
from typing import List, Dict, Any
import anthropic
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def generate_training_data(
    client: anthropic.Anthropic,
    num_examples: int,
    prompt_template: str
) -> List[Dict[str, str]]:
    """
    Generate training examples using Anthropic's Claude model.

    Args:
        client: Anthropic client instance
        num_examples: Number of examples to generate
        prompt_template: Template for generating examples

    Returns:
        List of dictionaries containing input-output pairs
    """
    examples = []
    batch_size = 5  # Process examples in small batches to handle rate limits

    for i in range(0, num_examples, batch_size):
        current_batch = min(batch_size, num_examples - i)
        print(f"Generating batch {i//batch_size + 1}/{(num_examples + batch_size - 1)//batch_size}")

        for _ in range(current_batch):
            try:
                # Generate a single example
                response = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=2000,
                    temperature=0.7,
                    system="You are a helpful AI assistant that generates high-quality Python code examples with documentation.",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_template
                        }
                    ]
                )

                # Extract the response content
                content = response.content[0].text

                # Parse the JSON response
                try:
                    example = json.loads(content)
                    if "code" in example and "documentation" in example:
                        # Create the training example
                        training_example = {
                            "input": example["code"],
                            "output": example["documentation"]
                        }
                        examples.append(training_example)
                    else:
                        print("Warning: Invalid example format, skipping")
                except json.JSONDecodeError:
                    print("Warning: Failed to parse JSON response, skipping")

                # Add a small delay to avoid rate limits
                time.sleep(1)

            except Exception as e:
                print(f"Error generating example: {str(e)}")
                continue

    print(f"Successfully generated {len(examples)} examples")
    return examples

def create_dataset(examples: List[Dict[str, str]]) -> Dataset:
    """
    Convert the generated examples into a HuggingFace Dataset.

    Args:
        examples: List of input-output pairs

    Returns:
        HuggingFace Dataset object
    """
    # Format the examples for training
    formatted_examples = []

    for example in examples:
        # Create a formatted prompt for the model
        formatted_prompt = f"""Below is a Python code snippet. Please generate comprehensive documentation for it.

Code:
{example['input']}

Documentation:
{example['output']}"""

        formatted_examples.append({
            "text": formatted_prompt,
            "input": example["input"],
            "output": example["output"]
        })

    # Create the dataset
    dataset = Dataset.from_list(formatted_examples)

    # Split the dataset into train and validation sets
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Created dataset with {len(dataset['train'])} training examples and {len(dataset['test'])} validation examples")
    return dataset

def prepare_models(
    teacher_model_name: str = "claude-3-7-sonnet-20250219",  # or any other large model
    student_model_name: str = "google/gemma-3-1b-it"
) -> tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and prepare both teacher and student models for distillation.

    Args:
        teacher_model_name: Name of the teacher model
        student_model_name: Name of the student model

    Returns:
        Tuple of (teacher_model, student_model, tokenizer)
    """
    # Load tokenizer (we'll use the student's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        student_model_name,
        trust_remote_code=True,
        padding_side="right",
        truncation_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher model (frozen)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    teacher_model.eval()  # Set to evaluation mode

    # Load student model (trainable)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare student model for training
    student_model = prepare_model_for_kbit_training(student_model)

    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to student model
    student_model = get_peft_model(student_model, lora_config)
    student_model.gradient_checkpointing_enable()

    return teacher_model, student_model, tokenizer

def train_model(
    teacher_model: AutoModelForCausalLM,
    student_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    output_dir: str
) -> None:
    """
    Train the student model using knowledge distillation.

    Args:
        teacher_model: The teacher model (frozen)
        student_model: The student model to train
        tokenizer: Tokenizer for both models
        dataset: Training dataset
        output_dir: Directory to save the fine-tuned model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="tensorboard"
    )

    # Initialize distillation trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=2.0,  # Temperature for softening probability distributions
        alpha=0.5,  # Weight for distillation loss
        model=student_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train the model
    print("Starting distillation training...")
    trainer.train()

    # Save the final model
    print("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = trainer.evaluate()
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training completed. Model saved to {output_dir}")
    print(f"Final evaluation metrics: {metrics}")

def main():
    # Initialize Anthropic client for data generation
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Define the prompt template for code documentation generation
    prompt_template = """Please generate a Python code snippet and its corresponding documentation.
    The documentation should include:
    1. A clear description of what the code does
    2. Explanation of key functions and their parameters
    3. Example usage
    4. Any important notes or limitations

    Format the response as a JSON object with two fields:
    - "code": The Python code snippet
    - "documentation": The complete documentation

    The code should be non-trivial but not too complex, suitable for a small model to learn from.
    Focus on common programming patterns and best practices."""

    # Generate training data
    examples = generate_training_data(
        client=client,
        num_examples=1000,
        prompt_template=prompt_template
    )

    # Create dataset
    dataset = create_dataset(examples)

    # Prepare models and tokenizer
    teacher_model, student_model, tokenizer = prepare_models()

    # Train model using distillation
    train_model(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir="./distilled_model"
    )

if __name__ == "__main__":
    main()
