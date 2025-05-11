import os
import json
from typing import List, Dict, Any
import anthropic
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add memory management settings
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def generate_single_example(client: anthropic.Anthropic) -> str:
    """
    Generate a single Python code example using Claude.

    Args:
        client: Anthropic client instance

    Returns:
        Python code example as string or None if generation fails
    """
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0.7,
            system="You are a helpful AI assistant that generates high-quality Python code examples.",
            messages=[{
                "role": "user",
                "content": "Generate a non-trivial but not too complex Python code example. Focus on common programming patterns and best practices. Return only the code, no documentation or explanation."
            }]
        )

        content = response.content[0].text.strip()
        # Remove any markdown code block formatting
        if content.startswith("```python"):
            content = content[9:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
    except Exception as e:
        print(f"Error generating example: {str(e)}")
        return None

def generate_training_data(
    client: anthropic.Anthropic,
    num_examples: int
) -> List[str]:
    """
    Generate Python code examples using Anthropic's Claude model.

    Args:
        client: Anthropic client instance
        num_examples: Number of examples to generate

    Returns:
        List of Python code examples
    """
    examples = []
    batch_size = 5
    total_batches = (num_examples + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        print(f"Generating batch {batch_num + 1}/{total_batches}")

        # Generate batch of examples
        batch_examples = []
        for _ in range(min(batch_size, num_examples - len(examples))):
            example = generate_single_example(client)
            if example:
                batch_examples.append(example)
            time.sleep(1)  # Rate limiting

        examples.extend(batch_examples)

        if len(examples) >= num_examples:
            break

    print(f"Successfully generated {len(examples)} examples")
    return examples

def create_dataset(examples: List[str]) -> Dataset:
    """
    Convert the generated examples into a HuggingFace Dataset.

    Args:
        examples: List of Python code examples

    Returns:
        HuggingFace Dataset object
    """
    # Format the examples for training
    formatted_examples = []

    for code in examples:
        # Create a formatted prompt for the model
        formatted_prompt = f"""Below is a Python code snippet. Please generate comprehensive documentation for it.

Code:
{code}

Documentation:"""

        formatted_examples.append({
            "text": formatted_prompt,
            "input": code
        })

    # Create the dataset
    dataset = Dataset.from_list(formatted_examples)

    # Split the dataset into train and validation sets
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Created dataset with {len(dataset['train'])} training examples and {len(dataset['test'])} validation examples")
    return dataset

class DistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_model: AutoModelForCausalLM,
        temperature: float = 2.0,
        alpha: float = 0.5,  # Weight for distillation loss
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get student model outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Move all inputs to CPU for teacher model
        cpu_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                cpu_inputs[k] = v.cpu()
            else:
                cpu_inputs[k] = v

        # Get teacher model outputs (no gradient computation)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**cpu_inputs)
            teacher_logits = teacher_outputs.logits

        # Ensure both logits are on the same device (GPU)
        teacher_logits = teacher_logits.to(student_logits.device)

        # Calculate task loss (language modeling loss)
        task_loss = student_outputs.loss

        # Calculate distillation loss (KL divergence)
        # Scale logits by temperature
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature

        # Calculate KL divergence loss
        distillation_loss = torch.nn.functional.kl_div(
            torch.log_softmax(student_logits, dim=-1),
            torch.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)  # Scale back

        # Combine losses
        loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss

        return (loss, student_outputs) if return_outputs else loss

def prepare_models(
    teacher_model_name: str = "google/gemma-3-12b-it",  # or any other large model
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

    # Load teacher model on CPU
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Keep teacher on CPU
        trust_remote_code=True
    )
    teacher_model.eval()  # Set to evaluation mode

    # Load student model on GPU
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
        eval_strategy="steps",
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
    try:
        from google.colab import userdata
        ANTHROPIC_API_KEY = userdata.get('ANTHROPIC_API_KEY')
    except ImportError:
        from dotenv import load_dotenv
        load_dotenv("examples/saved_keys.env")
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Generate training data
    examples = generate_training_data(
        client=client,
        num_examples=5  # TODO: Change to 1000
    )

    # Create dataset
    dataset = create_dataset(examples)

    # # Prepare models and tokenizer
    # teacher_model, student_model, tokenizer = prepare_models()

    # # Train model using distillation
    # train_model(
    #     teacher_model=teacher_model,
    #     student_model=student_model,
    #     tokenizer=tokenizer,
    #     dataset=dataset,
    #     output_dir="./distilled_model"
    # )

if __name__ == "__main__":
    main()
