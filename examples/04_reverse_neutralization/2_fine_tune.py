#!/usr/bin/env python3
"""
Fine-tuning script for reverse neutralization of email tone.

This script neutralizes emails and creates a fine-tuning dataset to train a model
that can convert neutral emails back to personalized ones.
"""

import json
import os
import time
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm


class EmailFineTuner:
    """
    A class for fine-tuning OpenAI models to convert neutral emails to personalized ones.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the EmailFineTuner.

        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model: Model to use for neutralization (default: gpt-4o-mini)
        """
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("Valid OpenAI API key is required")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.neutralization_prompt = """
Neutralize the tone and style from the following email to make it professional and suitable for communication between executives who may not know each other very well.

{email}
"""

    def load_emails(self, file_path: str) -> List[str]:
        """
        Load emails from a JSONL file.

        Args:
            file_path: Path to the JSONL file containing emails

        Returns:
            List of email strings
        """
        emails = []
        try:
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    # Handle both string emails and dict with email field
                    if isinstance(data, str):
                        emails.append(data)
                    elif isinstance(data, dict) and 'email' in data:
                        emails.append(data['email'])
                    else:
                        emails.append(str(data))
        except FileNotFoundError:
            raise FileNotFoundError(f"Email file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")

        return emails

    def neutralize_emails(self, emails: List[str]) -> List[str]:
        """
        Neutralize a list of emails using the OpenAI API.

        Args:
            emails: List of email strings to neutralize

        Returns:
            List of neutralized email strings
        """
        neutralized_emails = []

        for email in tqdm(emails, desc="Neutralizing emails"):
            prompt_with_email = self.neutralization_prompt.format(email=email)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_with_email}]
                )
                neutralized_emails.append(response.choices[0].message.content)
            except Exception as e:
                print(f"Error neutralizing email: {e}")
                neutralized_emails.append(email)  # Use original if neutralization fails

        return neutralized_emails

    def create_fine_tuning_dataset(self,
                                 original_emails: List[str],
                                 neutralized_emails: List[str],
                                 system_prompt: str = "You are a helpful assistant converting the neutralized email into personalized email.") -> List[Dict[str, Any]]:
        """
        Create a fine-tuning dataset from original and neutralized emails.

        Args:
            original_emails: List of original personalized emails
            neutralized_emails: List of neutralized emails
            system_prompt: System prompt for the fine-tuning dataset

        Returns:
            List of fine-tuning examples
        """
        dataset = []

        for original, neutralized in zip(original_emails, neutralized_emails):
            dataset.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": neutralized},
                    {"role": "assistant", "content": original}
                ]
            })

        return dataset

    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save the fine-tuning dataset to a JSONL file.

        Args:
            dataset: Fine-tuning dataset
            output_path: Path to save the dataset
        """
        with open(output_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")

        print(f"Dataset saved to {output_path}")

    def upload_training_file(self, file_path: str) -> str:
        """
        Upload a training file to OpenAI.

        Args:
            file_path: Path to the training file

        Returns:
            File ID of the uploaded file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training file not found: {file_path}")

        try:
            training_file = self.client.files.create(
                file=open(file_path, "rb"),
                purpose="fine-tune"
            )

            print(f"File uploaded successfully!")
            print(f"File ID: {training_file.id}")
            print(f"File status: {training_file.status}")

            # Wait for file to be processed
            time.sleep(5)

            return training_file.id

        except Exception as e:
            raise Exception(f"Error uploading file: {e}")

    def create_fine_tuning_job(self, training_file_id: str, base_model: str = "gpt-3.5-turbo") -> str:
        """
        Create a fine-tuning job.

        Args:
            training_file_id: ID of the uploaded training file
            base_model: Base model to fine-tune

        Returns:
            Job ID of the fine-tuning job
        """
        try:
            job = self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=base_model
            )

            print(f"Fine-tuning job created: {job.id}")
            return job.id

        except Exception as e:
            raise Exception(f"Error creating fine-tuning job: {e}")

    def monitor_fine_tuning_job(self, job_id: str, check_interval: int = 120) -> str:
        """
        Monitor a fine-tuning job until completion.

        Args:
            job_id: ID of the fine-tuning job
            check_interval: Time in seconds between status checks

        Returns:
            Fine-tuned model ID if successful
        """
        while True:
            job_status = self.client.fine_tuning.jobs.retrieve(job_id)
            print(f"Job status: {job_status.status}")

            if job_status.status == 'succeeded':
                print(f"Fine-tuning complete! Model: {job_status.fine_tuned_model}")
                return job_status.fine_tuned_model
            elif job_status.status == 'failed':
                raise Exception("Fine-tuning failed. Check the job status for more information.")

            print(f"Waiting {check_interval} seconds...")
            time.sleep(check_interval)

    def test_fine_tuned_model(self, model_id: str, test_email: str,
                            system_prompt: str = "You are a helpful assistant converting the neutralized email into personalized email.") -> str:
        """
        Test the fine-tuned model with a sample email.

        Args:
            model_id: ID of the fine-tuned model
            test_email: Test email to convert
            system_prompt: System prompt for the test

        Returns:
            Generated personalized email
        """
        try:
            completion = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test_email}
                ]
            )

            return completion.choices[0].message.content

        except Exception as e:
            raise Exception(f"Error testing fine-tuned model: {e}")

    def run_full_pipeline(self,
                         input_file: str,
                         output_dataset: str = "dataset.jsonl",
                         base_model: str = "gpt-3.5-turbo") -> str:
        """
        Run the full fine-tuning pipeline.

        Args:
            input_file: Path to input emails file
            output_dataset: Path to save the dataset
            base_model: Base model to fine-tune

        Returns:
            Fine-tuned model ID
        """
        print("=== Starting Fine-tuning Pipeline ===")

        # Load emails
        print("1. Loading emails...")
        emails = self.load_emails(input_file)
        print(f"Loaded {len(emails)} emails")

        # Neutralize emails
        print("2. Neutralizing emails...")
        neutralized_emails = self.neutralize_emails(emails)

        # Create dataset
        print("3. Creating fine-tuning dataset...")
        dataset = self.create_fine_tuning_dataset(emails, neutralized_emails)

        # Save dataset
        print("4. Saving dataset...")
        self.save_dataset(dataset, output_dataset)

        # Upload training file
        print("5. Uploading training file...")
        training_file_id = self.upload_training_file(output_dataset)

        # Create fine-tuning job
        print("6. Creating fine-tuning job...")
        job_id = self.create_fine_tuning_job(training_file_id, base_model)

        # Monitor job
        print("7. Monitoring fine-tuning job...")
        model_id = self.monitor_fine_tuning_job(job_id)

        print("=== Pipeline Complete ===")
        return model_id


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Fine-tune a model for email tone conversion")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file with emails")
    parser.add_argument("--output", "-o", default="dataset.jsonl", help="Output dataset file")
    parser.add_argument("--model", "-m", default="gpt-3.5-turbo", help="Base model to fine-tune")
    parser.add_argument("--neutralization-model", default="gpt-4o-mini", help="Model for neutralization")
    parser.add_argument("--env-file", default="../keys.env", help="Environment file with API keys")
    parser.add_argument("--test-email", help="Test email for the fine-tuned model")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file)

    try:
        # Initialize fine-tuner
        fine_tuner = EmailFineTuner(model=args.neutralization_model)

        # Run pipeline
        model_id = fine_tuner.run_full_pipeline(
            input_file=args.input,
            output_dataset=args.output,
            base_model=args.model
        )

        # Test the model if test email provided
        if args.test_email:
            print("\n=== Testing Fine-tuned Model ===")
            result = fine_tuner.test_fine_tuned_model(model_id, args.test_email)
            print("Generated personalized email:")
            print(result)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())



