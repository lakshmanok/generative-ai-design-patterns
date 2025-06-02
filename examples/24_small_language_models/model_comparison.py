# Fix for transformers compatibility issue with Gemma-3 models
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

import torch
# Disable TorchDynamo compilation globally to avoid unsupported operator errors
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch._dynamo as dynamo
dynamo.disable()
dynamo.config.suppress_errors = True
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
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
    print(f"Testing {model_name}...")
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad token is set for proper generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with additional safeguards
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
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

    # except Exception as e:
    except KeyError as e:
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
    model_names = ("google/gemma-3-1b-it", "google/gemma-3-12b-it")

    code_example = """
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Task:
    id: int
    title: str
    description: str
    completed: bool = False
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def mark_completed(self) -> None:
        self.completed = True

    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)


class TaskManager:
    def __init__(self, storage_path: Optional[Path] = None):
        self.tasks: Dict[int, Task] = {}
        self.storage_path = storage_path or Path(\"tasks.json\")
        self.next_id = 1

    def add_task(self, title: str, description: str, tags: List[str] = None) -> Task:
        task = Task(id=self.next_id, title=title, description=description, tags=tags)
        self.tasks[task.id] = task
        self.next_id += 1
        logger.info(f\"Added task: {task.title} (ID: {task.id})\")
        return task

    def get_task(self, task_id: int) -> Optional[Task]:
        return self.tasks.get(task_id)

    def delete_task(self, task_id: int) -> bool:
        if task_id in self.tasks:
            task = self.tasks.pop(task_id)
            logger.info(f\"Deleted task: {task.title} (ID: {task.id})\")
            return True
        return False

    def list_tasks(self, include_completed: bool = True) -> List[Task]:
        if include_completed:
            return list(self.tasks.values())
        return [task for task in self.tasks.values() if not task.completed]

    def save(self) -> None:
        try:
            with open(self.storage_path, 'w') as f:
                task_dict = {
                    \"next_id\": self.next_id,
                    \"tasks\": {
                        str(task_id): {
                            \"id\": task.id,
                            \"title\": task.title,
                            \"description\": task.description,
                            \"completed\": task.completed,
                            \"tags\": task.tags
                        }
                        for task_id, task in self.tasks.items()
                    }
                }
                json.dump(task_dict, f, indent=2)
            logger.info(f\"Tasks saved to {self.storage_path}\")
        except Exception as e:
            logger.error(f\"Failed to save tasks: {e}\")

    def load(self) -> bool:
        if not self.storage_path.exists():
            logger.warning(f\"Storage file {self.storage_path} not found\")
            return False

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.next_id = data.get(\"next_id\", 1)
                tasks_data = data.get(\"tasks\", {})

                self.tasks = {}
                for _, task_data in tasks_data.items():
                    task = Task(
                        id=task_data[\"id\"],
                        title=task_data[\"title\"],
                        description=task_data[\"description\"],
                        completed=task_data[\"completed\"],
                        tags=task_data[\"tags\"]
                    )
                    self.tasks[task.id] = task

                logger.info(f\"Loaded {len(self.tasks)} tasks from {self.storage_path}\")
                return True
        except Exception as e:
            logger.error(f\"Failed to load tasks: {e}\")
            return False

    """

    # Complex prompt that might challenge smaller models
    complex_prompt = f"""Below is a Python code snippet. Please generate documentation for the code.

Code:
{code_example}

Documentation:"""

    run_model_comparison(model_names, complex_prompt)

if __name__ == "__main__":
    main()

