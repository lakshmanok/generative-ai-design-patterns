#!/usr/bin/env python3
"""
Test the fine-tuned model with a single email.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv("../saved_keys.env")

class ModelTester:
    """Test fine-tuned models for email personalization."""

    def __init__(self, api_key: str = None):
        """Initialize the model tester."""
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("Valid OpenAI API key is required")

        self.client = OpenAI(api_key=api_key)


    def test_model(self, model_id: str, test_email: str, system_prompt: str = None) -> str:
        """Test the fine-tuned model with a single email."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant converting the neutralized email into personalized email."

        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test_email}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"


def demo_testing():
    """Demonstrate the testing functionality."""
    print("Fine-Tuned Model Testing Demo")
    print("=" * 50)

    # Initialize tester
    tester = ModelTester()

    # Example model ID (replace with your actual fine-tuned model)
    # You would get this from running fine_tune.py
    model_id = input("Enter your fine-tuned model ID (e.g., ft:gpt-3.5-turbo-0125:...): ")

    if not model_id:
        print("No model ID provided. Using gpt-3.5-turbo for demo...")
        model_id = "gpt-3.5-turbo"

    print(f"\nTesting model: {model_id}")
    print("-" * 40)

    # Test with a single email
    test_email = """Subject: Invitation to Present on Marketing Campaign for 2026 FIFA World Cup

Dear Gretl,

I hope this message finds you well. I am writing to officially invite you to give a presentation on the marketing campaign surrounding the 2026 FIFA World Cup. Your expertise and insights would be invaluable to our team, and we are eager to hear your thoughts on this exciting project.

We believe that your unique perspective and experience will bring a fresh and innovative approach to our marketing strategies for this upcoming event. Your presentation will provide valuable insights that will help guide our team in creating a successful campaign.

Please let me know at your earliest convenience if you are available and willing to present. We are looking forward to hearing from you and are excited about the opportunity to collaborate on this important project.

Thank you in advance for considering our invitation. We appreciate your time and expertise.

Warm regards,

[Your Name]
[Your Title]
[Company Name]

"""

    print("Test Email:")
    print(test_email)
    print("\nPersonalized Output:")

    result = tester.test_model(model_id, test_email)
    print(result)


if __name__ == "__main__":
    demo_testing()
