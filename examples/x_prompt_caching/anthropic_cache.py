import os
import time
from anthropic import Anthropic

from dotenv import load_dotenv

if os.path.exists("examples/keys.env"):
    load_dotenv("examples/keys.env")
else:
    raise FileNotFoundError("examples/keys.env not found")

def main():
    # Initialize the Anthropic client
    client = Anthropic()

    # Example prompt
    prompt = "Explain the concept of prompt caching in AI systems. Provide at least 1000 words."

    # First call - will make API request
    print("\nFirst call:")
    response1_start_time = time.time()
    response1 = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=4096,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]}]
    )
    response1_end_time = time.time()
    print(response1.content[0].text)

    # Second call - will use Anthropic's built-in caching
    print("\nSecond call:")
    response2_start_time = time.time()
    response2 = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=4096,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]}]
    )
    response2_end_time = time.time()
    print(response2.content[0].text)

    # Verify responses are identical
    print("\nResponses are identical:", response1.content[0].text == response2.content[0].text)

    print(f"Response time: {response1_end_time - response1_start_time:.2f} seconds")
    print(f"Response time: {response2_end_time - response2_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
